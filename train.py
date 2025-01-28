import torch
from actor import actor
from torch.utils.data import DataLoader
from critic import critic
from data_generator import data_generator
from state_transition import state_transition

torch.manual_seed(2)
epoch = 20
city = 20
batch = 512
instances = 2500
mse_loss = torch.nn.MSELoss()
actor = actor(4, 256)
critic = critic(4, 256)
LR = 0.0001
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
steps_per_epoch = 20
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
actor = actor.to(device)
critic = critic.to(device)
# After model creation

for param in actor.parameters():
    if len(param.shape) > 1:
        torch.nn.init.kaiming_normal_(param)
for param in critic.parameters():
    if len(param.shape) > 1:
        torch.nn.init.kaiming_normal_(param)
torch.set_printoptions(profile="full")
for e in range(epoch):
    for t in range(steps_per_epoch):
        batch = data_generator(city, 1000, 10)
        for state in batch:
            state = state.to(device)
            n_city = 0
            tour = []
            prev_action = None
            rewards = []
            lls = []
            state_value = critic(state)
            x0 = None
            while n_city < city:
                action, ll, x0 = actor(state, prev_node=prev_action, x0=x0)
                tour.append(action)
                lls.append(ll)
                state = state_transition(state, action, prev_action)
                prev_action = action
                n_city += 1
                if n_city == city:
                    path = torch.stack(tour, dim=1)
                    state_actor = state.x[path][:, :, 0:2]
                    distance = torch.sum(
                        torch.linalg.vector_norm(
                            state_actor[:, :-1, :] - state_actor[:, 1:, :], dim=2
                        ),
                        dim=1,
                    ) + torch.linalg.vector_norm(
                        state_actor[:, 0, :] - state_actor[:, -1, :], dim=1
                    )
                    lls = torch.stack(lls, dim=1)
                    lls = torch.sum(lls, dim=1)
                    actor_loss = (
                        (distance.detach() - state_value.detach()) * lls
                    ).mean()
                    critic_loss = mse_loss(
                        distance.detach().squeeze(), state_value.squeeze()
                    ).mean()
                    loss = critic_loss + actor_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_norm=1, norm_type=2
                    )
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_norm=1, norm_type=2
                    )
                    optimizer.step()

            print(
                distance.mean(dim=0),
                "critic loss",
                critic_loss,
                "actor_loss",
                loss,
                "epcoh",
                e,
                t,
                "step",
                scheduler.get_last_lr(),
            )
    scheduler.step()
torch.save(actor.state_dict(), f"actor_{city}.pt")

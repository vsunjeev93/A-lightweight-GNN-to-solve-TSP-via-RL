import torch
from actor import actor
from state_transition import state_transition
from data_generator import data_generator
params={'Nx':3, # number of MHA layers in encoder sequentially
'embed':128, #embedding dimension
'heads':8}
city=20
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model=actor(4,128).to(device)
model.load_state_dict(torch.load(f'actor_{city}.pt'))
model.eval()
batch = data_generator(city, 512,1 )
torch.manual_seed(2)
distances=[]
for state in batch:
    with torch.no_grad():
        prev_action=None
        tour=[]
        n_city=0
        x0=None
        state=state.to(device)
        while n_city < city:
            action, ll,x0 = model(state, prev_node=prev_action,x0=x0)
            tour.append(action)
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
                distances.append(distance)
print(torch.cat(distances).mean())
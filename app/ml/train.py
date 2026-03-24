import torch
import torch.nn as nn
#
from torch.utils.data import Dataset, DataLoader
from encode import encode_game
from dataset import load_games
from model import ChessNet, save_model, load_model
#
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))) #a little fix for colab
#

############################
class ChessDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        s = self.samples[index]
        state = torch.tensor(s["state"],  dtype=torch.float32)
        action = torch.tensor(s["action"], dtype=torch.long)
        value = torch.tensor(s["value"],  dtype=torch.float32)
        return state, action, value
    
    #Not so much to talk about, just personal dataset class to work with
    #Something clos to dict I guess, or dict itself
############################


##############################################
def build_dataset(pgnFILE, max_games = 10000):
    print(f"{max_games} games are loading")
    games = load_games(pgnFILE, max_games)
    samples = []

    for i, game in enumerate(games):
        samples.extend(encode_game(game))
        if i % 2000 == 0:
            print(f"Encoded {i}/{len(games)} games")

    print(f"Dataset is ready with {len(samples)} in total")
    return ChessDataset(samples)
##############################################


#######################################################################################################
def train(pgnFILE, epochs=30, batch_size=512, max_games=10_000, lr=1e-3, save_path="weights/model.pt"):
    #Machine Learning itself
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(pgnFILE, max_games)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = ChessNet().to(device)

    policy_loss_fn = nn.CrossEntropyLoss() #move prediction
    value_loss_fn  = nn.MSELoss() #winner prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #optimization idk

    for epoch in range(epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0

        for states, actions, values in loader:
            states  = states.to(device)
            actions = actions.to(device)
            values  = values.to(device).unsqueeze(1)  #(batch,) → (batch, 1)

            pred_policy, pred_value = model(states)

            p_loss = policy_loss_fn(pred_policy, actions)
            v_loss = value_loss_fn(pred_value, values)

            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += p_loss.item()
            total_value_loss  += v_loss.item()

        batches = len(loader)
        print(f"Epoch {epoch+1}/{epochs}"
              f"Policy loss: {total_policy_loss/batches:.4f}"
              f"Value loss: {total_value_loss/batches:.4f}")

        save_model(model, save_path)
        print(f"Saved in {save_path}")
#######################################################################################################


###########################
def train_by_chunk(pgnFILE, 
    chunks=100, games_per_chunk=15_000, epochs_per_chunk=15,
    batch_size=512, lr=1e-3, save_path="weights/model.pt"):
    #Dataset has 12M games, 15,000 * 100 = 1,500,00 games

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train existing model.pt
    if os.path.exists(save_path):
        model = load_model(save_path)
    #creates new model.pt if there's no
    else:
        model = ChessNet()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()

    for idx in range(chunks):
        print(f'Chunk {idx} is loading')
        offset = idx * games_per_chunk

        games  = load_games(pgnFILE, max_games=games_per_chunk, skip=offset)
        if not games: #finish
            print('Finished!')
            break
        
        samples = []
        for i, game in enumerate(games):
            samples.extend(encode_game(game))
            if i % 2000 == 0:
                print(f"  Encoded {i}/{len(games)} games")

        dataset = ChessDataset(samples)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        #training every chunk epochs_per_chunk times
        for epoch in range(epochs_per_chunk):
            model.train()
            total_p, total_v = 0.0, 0.0

            #state, action and value
            for st, act, val in loader:
                st  = st.to(device)
                act = act.to(device)
                val  = val.to(device).unsqueeze(1)
                pred_policy, pred_value = model(st)
                p_loss = policy_loss(pred_policy, act)
                v_loss = value_loss(pred_value, val)
                loss   = p_loss + v_loss

                #optimization and backtracking
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()

            batches = len(loader)
            print(f'Epoch {epoch + 1}')
            print(f"policy: {total_p/batches:.4f} | value: {total_v/batches:.4f}")

        save_model(model, save_path)
###########################


##########################
if __name__ == "__main__":
    train("data/lichess.pgn.zst")
##########################
import torch
import torch.nn as nn

# Q-Network
class CombatModel(nn.Module):
    def __init__(self, action_size):
        super(CombatModel, self).__init__()
        self.hand = DeckModel(10,256)
        self.draw = DeckModel(50,256)
        self.discard = DeckModel(75,32)
        self.exhaust = DeckModel(50,16)
        self.fc1 = nn.Linear(38110,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,action_size)


    def forward(self, inputs):
        hand_tensor, draw_tensor, discard_tensor, exhaust_tensor, combat_player_state_tensor, enemies_tensor = inputs
        hand = self.hand(hand_tensor)
        draw = self.draw(draw_tensor)
        discard = self.discard(discard_tensor)
        exhaust = self.exhaust(exhaust_tensor)
        if combat_player_state_tensor.dim() == 1:
            combat_player_state_tensor = combat_player_state_tensor.unsqueeze(0)
        if enemies_tensor.dim() == 1:
            enemies_tensor = enemies_tensor.unsqueeze(0)
        #decks = torch.cat((hand, combat_player_state_tensor, enemies_tensor),dim=1)
        decks = torch.cat((hand,draw,discard,exhaust, combat_player_state_tensor, enemies_tensor),dim=1)
        x = torch.relu(self.fc1(decks))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DeckModel(nn.Module):
    def __init__(self, deck_size, out_features):
        super(DeckModel, self).__init__()
        self.Conv1d = nn.Conv1d(in_channels=deck_size, out_channels=out_features, kernel_size=1)

    def forward(self,x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.Conv1d(x)
        return x.flatten(start_dim=1)
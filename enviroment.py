import warnings
import torch
import requests
import random
import time
from tensorgenerator import TensorGenerator

class enviroment:
    HOST_URL = "http://127.0.0.1:5000"
    PLAY_GAME_PATH = "play_game"
    FULL_GAME_URL = f"{HOST_URL}/{PLAY_GAME_PATH}"
    headers = {'Content-Type': 'application/json'}


    def __init__(self):
        self.TensorGenerator = TensorGenerator()
        self.state_size = 12602
        self.action_size = 176
        self.history = {'cumulative_reward': 0,'cumulative_floor_reward': 0,'cumulative_damage_reward': 0,
                        'cumulative_hp_reward' : 0}
        self.last_response = None
        self.start_game()
        self.continue_previous_action = False
        self.previous_action = None
        self.previous_move = None
        self.reached_floor = 1
        warnings.filterwarnings('ignore')

    def start_game(self):
        self.last_response = requests.post(self.FULL_GAME_URL, headers=self.headers, json={}).json()
        combat_state = self.combat_state2tensor()
        non_combat_state = self.noncombat_state2tensor()
        self.history = {'cumulative_reward': 0,'cumulative_floor_reward': 0,'cumulative_damage_reward': 0,
                        'cumulative_hp_reward' : 0}
        return combat_state, non_combat_state

    def reset(self):
        self.start_game()

    def on_game_over(self):
        #print(f"Reached floor {self.last_response['game']['game_state']['floor_num']}")
        if 'game_over' in self.last_response:
            if self.last_response['game_over'] == 'True':
                print(f"Won with {self.get_current_hp()} hp")
        pass

    def in_combat(self):
        return 'enemy_list' in self.last_response['game']['floor']

    def step(self, action_idx):
        hand_ids = self.get_hand_ids()
        enemy_ids = self.get_enemy_ids()
        discard_ids = self.get_discard()
        potion_ids = self.get_potion_ids()
        prev_response = self.last_response
        action = self.TensorGenerator.tensor2move(action_idx,hand_ids,enemy_ids,discard_ids, potion_ids)
        options_list = self.last_response['options']
        include = True

        def is_multi_move(potential_action):
            if len(action[0].split()) < 2:
                return False, None
            card = action[0].split()[1].split('_')
            if card[0] in ['warcry','headbutt','burning']:
                if card[1] == 'pact':  #burning pact
                    return True, 'burning_pact'
                elif card[0] == 'warcry':
                    return True, 'warcry'
                elif card[0] == 'headbutt':
                    if len(self.get_discard_ids()) > 0:
                        return True, 'headbutt'
                    else:
                        return False, None
                elif card[0] == 'dual':
                    return True, 'dual_wield'

            return False, None

        multi_move, move = is_multi_move(action[0])
        if multi_move:
            self.continue_previous_action = True
            self.previous_move = move
            self.previous_action = action[0]
            return self.combat_state2tensor(), 0, False, True
        else:
            if self.continue_previous_action:
                if self.previous_move in ['warcry', 'dual_wield', 'burning_pact']:
                    action[0] = self.previous_action + action[0]
                elif self.previous_move in ['headbutt']:
                    potential_action_split = self.previous_action.split()
                    potential_action = (
                            potential_action_split[0] + ' ' + potential_action_split[1] + ' '  # play headbutt_123
                            + action[0] + ' ' + potential_action_split[2])
                    if potential_action in options_list:
                        action[0] = potential_action


            self.continue_previous_action = False
            self.previous_move = None


        if action[0] in options_list:
            include = True
        else:
            warnings.warn(f'Action: {action} not in options_list {options_list}. Choosing random action.')
            action = self.select_random_action()
            include = False



        self.last_response = requests.post(self.FULL_GAME_URL, json={"action": action}, headers=self.headers).json()
        if 'error' in self.last_response:
            warnings.warn(f"action {action} error {options_list} choosing random action")
            action = self.select_random_action()
            self.last_response = requests.post(self.FULL_GAME_URL, json={"action": action}, headers=self.headers).json()
            include = False

        #print(f"taking action {action[0]}")
        reward = self.calculate_reward(prev_response)

        done = self.get_current_hp() <= 0 or 'game_over' in self.last_response
        if done:
            self.on_game_over()

        if self.in_combat():
            return self.combat_state2tensor(), reward, done, include
        else:
            warnings.warn("Non-combat not yet implemented")
            return None, 0, False, False

    def get_hand_ids(self):
        return list(self.get_hand().keys())

    def get_discard_ids(self):
        return list(self.get_discard().keys())

    def get_enemy_ids(self):
        return [enemy['id'] for enemy in self.get_enemies()]

    def valid_moves_mask(self):
        options_list = self.last_response["options"]
        if self.continue_previous_action:
            mask = torch.ones(self.action_size, dtype=torch.bool)
            if self.previous_move == 'warcry':
                hand = self.get_hand_ids()
                for i in range(len(hand)):
                    card = hand[i]
                    potential_action = self.previous_action + card
                    if potential_action in options_list:
                        mask[67 + i] = False
            elif self.previous_move == 'burning_pact':
                hand = self.get_hand_ids()
                for i in range(len(hand)):
                    card = hand[i]
                    potential_action = self.previous_action + card
                    if potential_action in options_list:
                        mask[77 + i] = False
            elif self.previous_move == 'dual_wield':
                hand = self.get_hand_ids()
                for i in range(len(hand)):
                    card = hand[i]
                    potential_action = self.previous_action + card
                    if potential_action in options_list:
                        mask[77 + i] = False
            elif self.previous_move == 'headbutt':
                discard = self.get_discard_ids()
                for i in range(len(discard)):
                    card = discard[i]
                    potential_action_split = self.previous_action.split()
                    if len(potential_action_split) > 3:
                        potential_action = (potential_action_split[0] + ' ' + potential_action_split[
                            1] + ' '  # play headbutt_123
                                            + card + ' ' + potential_action_split[3])  # strike_456 enemy_789
                        if potential_action in options_list:
                            mask[101 + i] = False
            else:
                raise Exception(f"Error {self.previous_action}")
            mask[0] = True
            return mask
        hand_ids = self.get_hand_ids()
        enemy_ids = self.get_enemy_ids()
        discard_ids = self.get_discard_ids()
        potion_ids = self.get_potion_ids()
        mask = torch.ones(self.action_size, dtype=torch.bool)
        for option in options_list:
            index = self.TensorGenerator.move2tensor(option, hand_ids, enemy_ids, discard_ids, potion_ids)
            parsed_move = self.TensorGenerator.tensor2move(index, hand_ids,enemy_ids,discard_ids,potion_ids)
            if parsed_move[0] in options_list:
                mask[index] = False
            else:
                if option.split()[1].split('_')[0] in ['warcry','dual','burning']:
                    mask[index] = False
                elif option.split()[1].split('_')[0] in ['headbutt']:
                    if len(option.split()) == 3:
                        mask[index] = True
                    else:
                        mask[index] = False
                else:
                    warnings.warn(f"Move: {option} not parsed correctly. {parsed_move[0]} not in {options_list}")

        return mask

    def combat_state2tensor(self):
        hand_tensor, draw_tensor, discard_tensor, exhaust_tensor = self.combat_player_cards2tensor()
        combat_player_state_tensor = self.combat_player_state2tensor()
        enemies_tensor = self.combat_enemies2tensor()
        return (hand_tensor, draw_tensor, discard_tensor, exhaust_tensor, combat_player_state_tensor, enemies_tensor)

    def select_random_action(self):
        options_list = self.last_response["options"]
        action = random.choice(options_list).strip()
        if not action == 'end_turn':
            return [action]
        elif options_list == ['end_turn']:
            return ['end_turn']
        else:
            return self.select_random_action()

    def make_random_action(self):
        action = self.select_random_action()
        self.last_response = requests.post(self.FULL_GAME_URL, json={"action": action}, headers=self.headers).json()

    def noncombat_state2tensor(self):
        max_hp = self.get_max_hp()
        current_hp = self.get_current_hp()
        return torch.tensor([max_hp, current_hp])

    def calculate_reward(self, prev_response):
        prev_hp = prev_response['game']['game_state']['player']['hp']
        current_hp = self.get_current_hp()
        delta_hp = current_hp - prev_hp
        floor_reward = 1 if 'rewards' in self.last_response['game']['floor'] else 0
        damage_dealt = self.calculate_damage_dealt(prev_response)
        #print(f"Rewards: delta_hp {delta_hp/ 10} floor reward {floor_reward} damage dealt {damage_dealt * 0.2}")
        reward = delta_hp / 10 + floor_reward #+ damage_dealt * 0.02
        self.history['cumulative_floor_reward'] += floor_reward
        #self.history['cumulative_damage_reward'] += damage_dealt * 0.2
        self.history['cumulative_hp_reward'] += delta_hp * 0.1
        self.history['cumulative_reward'] += reward
        return reward
        #return floor_reward

    def calculate_damage_dealt(self, prev_response):
        current_sum_hp = 0
        if 'enemy_list' in self.last_response['game']['floor']:
            enemies = self.get_enemies()
            for enemy in enemies:
                enemy_hp = enemy['hp']
                current_sum_hp += enemy_hp
        prev_sum_hp = 0
        if 'enemy_list' in prev_response['game']['floor']:
            prev_enemies = prev_response['game']['floor']['enemy_list']
            for enemy in prev_enemies:
                enemy_hp = enemy['hp']
                prev_sum_hp += enemy_hp
        return prev_sum_hp - current_sum_hp

    def combat_player_cards2tensor(self):
        """
        TO BE USED IN COMBAT ONLY
        :return: tensor representation of hand + draw + discard + exhaust
        """
        hand = self.get_hand()
        draw = self.get_draw()
        discard = self.get_discard()
        exhaust = self.get_exhaust()

        hand_tensor = self.TensorGenerator.deck2tensor(hand, 10) #num_cards in deck to be represented. Extra cards are cut off.
        draw_tensor = self.TensorGenerator.deck2tensor(draw, 50)
        discard_tensor = self.TensorGenerator.deck2tensor(discard, 75)
        exhaust_tensor = self.TensorGenerator.deck2tensor(exhaust, 50)


        return hand_tensor, draw_tensor, discard_tensor, exhaust_tensor

    def combat_player_state2tensor(self):
        """
        TO BE USED IN COMBAT ONLY
        Returns a tensor representation of a player's state in combat
        Currently implmenents: Max hp, current hp
        In development:str, dex, buffs/debuffs, relics
        :return: tensor representation of a player's combat state
        """
        max_hp = self.get_max_hp()
        current_hp = self.get_current_hp()
        block = self.get_current_block()
        energy = self.get_current_energy()
        floor = self.get_current_floor()
        return torch.tensor([max_hp / 100,current_hp / 100, block/10,energy/3, floor / 10])

    def combat_enemies2tensor(self):
        enemies = self.get_enemies()
        return self.TensorGenerator.enemies2tensor(enemies)

    def get_hand(self):
        game = self.last_response['game']
        floor = game['floor']
        player = floor['player']
        hand = player['hand']
        return hand

    def get_draw(self):
        game = self.last_response['game']
        floor = game['floor']
        player = floor['player']
        draw = player['draw_pile']
        return draw

    def get_discard(self):
        game = self.last_response['game']
        floor = game['floor']
        player = floor['player']
        discard = player['discard_pile']
        return discard

    def get_exhaust(self):
        game = self.last_response['game']
        floor = game['floor']
        player = floor['player']
        exhaust = player['exhaust_pile']
        return exhaust

    def get_relics(self):
        pass

    def get_max_hp(self):
        max_hp = self.last_response['game']['game_state']['player']['max_hp']
        return max_hp

    def get_current_hp(self):
        current_hp = self.last_response['game']['game_state']['player']['hp']
        return current_hp

    def get_enemies(self):
        game = self.last_response['game']
        floor = game['floor']
        enemies = floor['enemy_list']
        return enemies

    def get_current_energy(self):
        return self.last_response['game']['floor']['player']['energy']

    def get_current_block(self):
        if 'block' in self.last_response['game']['floor']['player']:
            return self.last_response['game']['floor']['player']['block']
        else:
            return 0

    def get_potion_ids(self):
        """
        FOR USE IN COMBAT ONLY
        :return: dict representing potions held by player
        """
        if 'potions' not in self.last_response['game']['floor']['player']:
            return []
        potions = self.last_response['game']['floor']['player']['potions']
        return list(potions.keys())

    def get_current_floor(self):
        return self.last_response['game']['game_state']['floor_num']

    def log(self, episode, epsilon):
        print(f"Episode: {episode} Epsilon: {epsilon}. Reached floor {self.last_response['game']['game_state']['floor_num']}. "
              f"Total reward: {self.history['cumulative_reward']} "
              f"hp_rewards: {self.history['cumulative_hp_reward']} "
              f"floor_reward: {self.history['cumulative_floor_reward']}")
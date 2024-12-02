
import numpy as np

class PTTTGame:
    def __init__(self, roots, infosets) -> None:
        self.n_players = 2
        self.roots = roots
        self.infosets = infosets

    def _build_board(self, infosets):
        board = [None for _ in range(9)]
        for i in range(len(infosets)):
            for j in range(1, len(infosets[i]), 2):
                action, outcome = infosets[i][j:j+2]
                if outcome == '*':
                    board[int(action)] = i
        return board

    def _terminal_rewards(self, board):
        for i in range(self.n_players):
            for j in range(3):
                if board[3*j + 0] == i and board[3*j + 1] == i and board[3*j + 2] == i:
                    if i == 0:
                        return True, [1, -1]
                    else:
                        return True, [-1, 1]
                if board[j + 3*0] == i and board[j + 3*1] == i and board[j + 3*2] == i:
                    if i == 0:
                        return True, [1, -1]
                    else:
                        return True, [-1, 1]
            
            if board[0] == i and board[4] == i and board[8] == i:
                if i == 0:
                    return True, [1, -1]
                else:
                    return True, [-1, 1]

            if board[2] == i and board[4] == i and board[6] == i:
                if i == 0:
                    return True, [1, -1]
                else:
                    return True, [-1, 1]
        
        for notch in board:
            if notch is None:
                return False, [0, 0]
        return True, [0, 0]

    def next_state(self, infosets, actions):
        infosets = infosets.copy()
        board = self._build_board(infosets)
        for i in range(len(actions)):
            infosets[i] += str(actions[i])
            if board[actions[i]] is None:
                infosets[i] += '*'
                board[actions[i]] = i
            else: 
                infosets[i] += "."
        
        terminal, rewards = self._terminal_rewards(board)

        return infosets, rewards, terminal

    def choose_uniform_action(self, infoset, player, sample=False):
        SAMPLE_SIZE = 3
        if sample:
            valid_actions = self.valid_actions(infoset, player)
            return np.random.choice(valid_actions, size=min(SAMPLE_SIZE, len(valid_actions)), replace=False), 1/len(valid_actions) * np.ones(min(SAMPLE_SIZE, len(valid_actions)))

        valid_actions = self.valid_actions(infoset, player)
        return np.random.choice(valid_actions), 1/len(valid_actions)

    def n_actions(self, infoset):
        return 9
    
    def valid_actions(self, infoset, player):
        valid_actions = []
        for i in range(9):
            if str(i) not in infoset:
                valid_actions.append(i)
        
        if player == 0: # to augment strategies, such that if opp has winning spot then block it immediately
            board = self._build_player_board(infoset, player)
            opp_spots = self._find_winning_spots(board, 1)
            player_spots = self._find_winning_spots(board, 0)
            if len(opp_spots) > 0:
                valid_actions = opp_spots
            elif len(player_spots) > 0:
                valid_actions = player_spots

        return np.array(valid_actions)

    def _build_player_board(self, infoset, player):
        opp = 1 - player
        board = [None for _ in range(9)]
        for i in range(1, len(infoset), 2):
            action, outcome = infoset[i:i+2]
            if outcome == '*':
                board[int(action)] = player
            else:
                board[int(action)] = opp
        return board
    
    def _find_winning_spots(self, board, player):
        spots = []
        for i in range(3):
            if board[3*i + 0] == player and board[3*i + 1] == player and board[3*i + 2] == None:
                spots.append(3*i + 2)
            elif board[3*i + 1] == player and board[3*i + 2] == player and board[3*i + 0] == None:
                spots.append(3*i + 0)
            elif board[3*i + 0] == player and board[3*i + 2] == player and board[3*i + 1] == None:
                spots.append(3*i + 1)
            elif board[i + 3*0] == player and board[i + 3*1] == player and board[i + 3*2] == None:
                spots.append(i + 3*2)
            elif board[i + 3*1] == player and board[i + 3*2] == player and board[i + 3*0] == None:
                spots.append(i + 3*0)
            elif board[i + 3*0] == player and board[i + 3*2] == player and board[i + 3*1] == None:
                spots.append(i + 3*1)
        
        if board[0] == player and board[4] == player and board[8] == None:
            spots.append(8)
        elif board[0] == player and board[8] == player and board[4] == None:
            spots.append(4)
        elif board[4] == player and board[8] == player and board[0] == None:
            spots.append(0)

        if board[2] == player and board[4] == player and board[6] == None:
            spots.append(6)
        elif board[2] == player and board[6] == player and board[4] == None:
            spots.append(4)
        elif board[4] == player and board[6] == player and board[2] == None:
            spots.append(2)

        return np.array(spots)

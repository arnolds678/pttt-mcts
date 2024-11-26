
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

    def choose_uniform_action(self, infoset, player):
        valid_actions = self.valid_actions(infoset, player)
        return np.random.choice(valid_actions), 1/len(valid_actions)

    def n_actions(self, infoset):
        return 9
    
    def valid_actions(self, infoset, player):
        valid_actions = []
        for i in range(9):
            if str(i) not in infoset:
                valid_actions.append(i)

        return np.array(valid_actions)
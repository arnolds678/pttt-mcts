
import numpy as np

class Node(object):
    def __init__(self, n, depth, n_actions, valid_actions) -> None:
        self.n = n
        self.depth = depth
        self.rewards = np.zeros(n_actions)
        self.rewards[valid_actions] = 1
        self.valid_actions = valid_actions
        self.update_probabilities()

        assert self.depth == 10 - len(valid_actions)

    def choose_action(self):
        p = self.probabilities / np.sum(self.probabilities)
        action = np.random.choice(len(p), p=p)
        return action, p[action]

    def update_reward(self, reward, action, prob):
        assert action in self.valid_actions

        self.rewards[action] *= np.exp(1.7**(self.depth - 9) * reward / prob)
        self.rewards /= np.sum(self.rewards)

        self.update_probabilities()
    
    def update_probabilities(self):
        # self.probabilities = self.rewards / np.sum(self.rewards)
        # self.probabilities *= (1 - self.n**(-0.3))
        self.probabilities = self.rewards * (1 - self.n**(-0.3))
        self.probabilities[self.valid_actions] += self.n**(-0.3) / len(self.valid_actions)
    
    def strategy_vector(self):
        return self.probabilities / np.sum(self.probabilities)

class MCTS(object):
    def __init__(self, game, num_iters) -> None:
        self.game = game
        self.num_iters = num_iters
        self.nodes = [{} for _ in range(len(game.roots))]

    def solve(self):
        for n in range(self.num_iters):
            self.simulate(n+1)

    def simulate(self, n):
        n_players = self.game.n_players
        in_tree = [True for _ in range(n_players)]
        curr_infosets = [root for root in self.game.roots]
        paths = [[] for _ in range(n_players)]
        terminal = False

        depth = 1
        while not terminal:
            actions = []
            for i in range(n_players):
                if not in_tree[i]:
                    action, prob = self.game.choose_uniform_action(curr_infosets[i], i)
                else:
                    if curr_infosets[i] in self.nodes[i]:
                        curr_node = self.nodes[i][curr_infosets[i]]
                        action, prob = curr_node.choose_action()
                    else:
                        action, prob = self.game.choose_uniform_action(curr_infosets[i], i)
                        curr_node = Node(n, depth, self.game.n_actions(curr_infosets[i]), self.game.valid_actions(curr_infosets[i], i))
                        self.nodes[i][curr_infosets[i]] = curr_node
                        in_tree[i] = False

                    paths[i].append((curr_infosets[i], action, prob))
                
                actions.append(action)
        
            curr_infosets, rewards, terminal = self.game.next_state(curr_infosets, actions)
            depth += 1

        for i in range(n_players):
            for infoset, action, prob in paths[i][::-1]:
                self.nodes[i][infoset].update_reward(rewards[i], action, prob)
    
    def get_strategy(self, player, infoset_txt_file):
        lines = open(infoset_txt_file).readlines()
        # Initially, set the tensor with all ones. We will mask out the illegal actions below,
        # and then we will normalize row-wise just before saving the tensor below.
        tensor = np.ones((len(lines), 9), dtype=np.float32)
        for idx, line in enumerate(lines):
            line = line.strip() # Remove the \n
            assert line.startswith('|')
            n = len(line) - 1
            assert n % 2 == 0
            n = n // 2

            if line in self.nodes[player]:
                tensor[idx] = self.nodes[player][line].strategy_vector()

            for j in range(n):
                pos = line[1 + 2 * j]
                outcome = line[2 + 2 * j]
                assert outcome in '*.'
                assert pos in '012345678'
                pos = int(pos)
                # Set zero probability to illegal actions. (Remember that we cannot probe a cell
                # more than once!)
                tensor[idx,pos] = 0

            if idx % 1000000 == 0:
                print(idx, 'done out of', len(lines))
        # Renormalize row wise
        tensor /= np.sum(tensor, axis=1)[:,None]
        np.save(f'pttt_pl{player}.npy', tensor)

            

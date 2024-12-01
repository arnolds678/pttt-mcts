
import numpy as np

SAMPLE_SIZE = 3
class Node(object):
    def __init__(self, n, depth, n_actions, valid_actions) -> None:
        self.n = n
        self.depth = depth
        self.rewards = np.zeros(n_actions)
        # self.rewards[valid_actions] = 1 # EXP3
        self.valid_actions = valid_actions
        self.visits = 0

        self.avg_probabilities = np.zeros(n_actions)
        self.avg_probabilities[valid_actions] = 1
        self.avg_probabilities /= len(valid_actions)
        self.counter = 0

        self.update_probabilities()
        # print(self.depth, self.valid_actions)
        # assert self.depth == len()#10 - len(valid_actions)

    def choose_action(self, sample=False):
        if sample:
            p = self.probabilities / np.sum(self.probabilities)
            actions = np.random.choice(len(p), size=min(SAMPLE_SIZE, len(self.valid_actions)), p=p, replace=False)
            return actions, p[actions]

        p = self.probabilities / np.sum(self.probabilities)
        action = np.random.choice(len(p), p=p)
        return action, p[action]

    def update_reward(self, reward, action, prob, n):
        assert (action in self.valid_actions, action, self.valid_actions)
        ## EXP-3
        # new_action_reward = self.rewards[action] * np.exp(1.7**(self.depth - 9) * reward / prob)
        # if new_action_reward != 0:
        #     self.rewards[action] = new_action_reward
        # # NOTE: when reward blows up to inf, make it deterministic by setting all other rewards to 0
        # if np.any(np.isposinf(self.rewards)):
        #     new_rewards = np.zeros_like(self.rewards, dtype=int) + 10 ** -200
        #     new_rewards[np.isposinf(self.rewards)] = 1 
        #     self.rewards = np.copy(new_rewards)
        self.n = n
        ## Regret-matching
        for a in self.valid_actions:
            if a != action:
                self.rewards[a] -= reward
            else:
                self.rewards[a] += reward / prob
        self.update_probabilities()
    
    def update_probabilities(self, test_param=None):
        pos_rewards = np.maximum(self.rewards, 0)
        if pos_rewards.sum() == 0:
            self.probabilities = np.zeros(9)
            self.probabilities[self.valid_actions] = 1
            self.probabilities / len(self.valid_actions)
        else:
            self.probabilities = pos_rewards / np.sum(pos_rewards)
        self.probabilities *= (1 - self.n**(-0.1))
        self.probabilities[self.valid_actions] += self.n**(-0.1) / len(self.valid_actions)

        self.avg_probabilities = (self.avg_probabilities * (self.counter + 1) + self.probabilities) / (self.counter + 2)
        self.counter += 1
    
    def strategy_vector(self):
        # return self.probabilities / np.sum(self.probabilities)
        return self.avg_probabilities

class MCTS(object):
    def __init__(self, game, num_iters) -> None:
        self.game = game
        self.num_iters = num_iters
        self.nodes = [{} for _ in range(len(game.roots))]

    def solve(self, sample=False):
        for n in range(self.num_iters):
            self.simulate(n+1, sample=sample)
            if n % 10000 == 0:
                print(n, 'done out of', self.num_iters)
    
    def traverse_tree(self, curr_infosets, in_tree, paths, n, depth, terminal):
        n_players = self.game.n_players
        actions = []
        add_to_path = [val for val in in_tree]
        for i in range(n_players):
            if not in_tree[i]:
                action_samples, prob_samples = self.game.choose_uniform_action(curr_infosets[i], i, sample=False)
                action_samples = [action_samples]
                prob_samples = [prob_samples]
            else:
                if curr_infosets[i] in self.nodes[i]:
                    curr_node = self.nodes[i][curr_infosets[i]]
                    action_samples, prob_samples = curr_node.choose_action(sample=True)
                else:
                    action_samples, prob_samples = self.game.choose_uniform_action(curr_infosets[i], i, sample=True)
                    assert depth == (len(curr_infosets[i]) - 1) // 2 + 1
                    curr_node = Node(n, depth, self.game.n_actions(curr_infosets[i]), self.game.valid_actions(curr_infosets[i], i))
                    self.nodes[i][curr_infosets[i]] = curr_node
                    in_tree[i] = False

                # paths[i].append((curr_infosets[i], action, prob))
            
            actions.append((action_samples, prob_samples))

        actions_0, probs_0 = actions[0]
        actions_1, probs_1 = actions[1]
        for i in range(len(actions_0)):
            action_0, prob_0 = actions_0[i], probs_0[i]

            if add_to_path[0]:
                paths[0].append((curr_infosets[0], action_0, prob_0))

            for j in range(len(actions_1)):
                action_1, prob_1 = actions_1[j], probs_1[j]

                if add_to_path[1]:
                    paths[1].append((curr_infosets[1], action_1, prob_1))

                # print('ITER:', n, 'DEPTH:', depth, 'action idx:', j, 'curr_infosets', curr_infosets)
                new_infosets, rewards, terminal = self.game.next_state(curr_infosets, [action_0, action_1])

                if terminal:
                    # print('FINAL PATHS:', paths, ' DEPTH:', depth, ' ITER:', n)
                    # print('FINAL infoset', new_infosets, 'REWARDS:', rewards)
                    for k in range(n_players):
                        for infoset, action, prob in paths[k][::-1]:
                            self.nodes[k][infoset].update_reward(rewards[k], action, prob)
                else:
                    # print('starting next traversal')
                    self.traverse_tree(new_infosets, in_tree, paths, n, depth+1, terminal)
                    # print('completed traversal')

                if add_to_path[1]:
                    paths[1] = paths[1][:-1]
            
            if add_to_path[0]:
                paths[0] = paths[0][:-1]

    def simulate(self, n, sample=False):
        # if sample: when simulating / traversing tree, sample multiple actions instead of just one per simulation
        n_players = self.game.n_players
        in_tree = [True for _ in range(n_players)]
        curr_infosets = [root for root in self.game.roots]
        paths = [[] for _ in range(n_players)]
        terminal = False
        depth = 1

        if sample:
            self.traverse_tree(curr_infosets, in_tree, paths, n, depth, terminal)
            return

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
                        curr_node.visits += 1
                    else:
                        action, prob = self.game.choose_uniform_action(curr_infosets[i], i)

                        assert depth == (len(curr_infosets[i]) - 1) // 2 + 1

                        curr_node = Node(n, depth, self.game.n_actions(curr_infosets[i]), self.game.valid_actions(curr_infosets[i], i))
                        self.nodes[i][curr_infosets[i]] = curr_node
                        in_tree[i] = False

                    paths[i].append((curr_infosets[i], action, prob))
                
                actions.append(action)
        
            curr_infosets, rewards, terminal = self.game.next_state(curr_infosets, actions)
            depth += 1

        for i in range(n_players):
            for infoset, action, prob in paths[i][::-1]:
                self.nodes[i][infoset].update_reward(rewards[i], action, prob, n)
        # print(self.nodes[0]['|'].rewards)
    
    def get_strategy(self, player, infoset_txt_file, alpha=None):
        # alpha controls how much to weigh strategies from already seen infosets

        lines = open(infoset_txt_file).readlines()
        # Initially, set the tensor with all ones. We will mask out the illegal actions below,
        # and then we will normalize row-wise just before saving the tensor below.

        print('preprocessing infosets') # groups all infosets with same set of * (success) and . (fail)
        if alpha is not None:
            mapping_infosets = {}
            reduced_infosets = {}
            for idx, line in enumerate(lines):
                line = line.strip()
                successful = []
                failed = []
                for i in range(1, len(line), 2):
                    if line[i+1] == '*':
                        successful.append(int(line[i]))
                    else:
                        failed.append(int(line[i]))
                successful.sort()
                failed.sort()
                mapping_infosets[line] = (tuple(successful), tuple(failed))
                reduced_infosets[(tuple(successful), tuple(failed))] = (np.zeros(9), 0)

                if idx % 1000000 == 0:
                    print(idx, 'done out of', len(lines))

        tensor = np.ones((len(lines), 9), dtype=np.float32)
        print('filling strategies for nodes found in tree')
        for idx, line in enumerate(lines):
            line = line.strip() # Remove the \n
            assert line.startswith('|')
            n = len(line) - 1
            assert n % 2 == 0
            n = n // 2

            if line in self.nodes[player]:
                if line == '|':
                    self.nodes[player][line].update_probabilities(test_param=4)
                    print("fdfdsfd", self.nodes[player][line].rewards, self.nodes[player][line].probabilities)
                count += 1
                tensor[idx] = self.nodes[player][line].strategy_vector()
                if alpha is not None:
                    strat, count = reduced_infosets[mapping_infosets[line]]
                    reduced_infosets[mapping_infosets[line]] = (strat + tensor[idx], count + 1)
            
            if idx % 1000000 == 0:
                print(idx, 'done out of', len(lines))
        
        print('filling strategies for nodes not found in tree')
        for idx, line in enumerate(lines):
            line = line.strip() # Remove the \n
            assert line.startswith('|')
            n = len(line) - 1
            assert n % 2 == 0
            n = n // 2

            if alpha is not None:
                if line not in self.nodes[player]:
                    strat, count = reduced_infosets[mapping_infosets[line]]
                    n_actions = len(self.game.valid_actions(line, player))
                    if count > 0 and n_actions > 0:
                        tensor[idx] = alpha * strat / count + (1-alpha) * 1/n_actions

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
        print(f'Lines changed for player {player}: {count}')
        np.save(f'pttt_pl{player}.npy', tensor)

            

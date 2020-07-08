class EpisodeStats:
    def __init__(self, episode_lengths, episode_rewards):
        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards

    def to_tuple(self):
        return (self.episode_lengths, self.episode_rewards)

import numpy as np

def get_binned_dists(n_per_group=7):
    """ Returns sub distributions from each of the 5 bins

        Args:
            n_per_group: number of distributions per bin.

    """

    n_groups = 4

    # Sample 1 million random sub distributions from the flat dirichlet.
    n_samples = int(1e6)
    random_dists = np.random.dirichlet(np.ones(n_groups), size=n_samples)

    groups = {}

    groups[0] = random_dists[(random_dists[:, 0] > 0.5)]
    groups[1] = random_dists[(random_dists[:, 1] > 0.5)]
    groups[2] = random_dists[(random_dists[:, 2] > 0.5)]
    groups[3] = random_dists[(random_dists[:, 3] > 0.5)]

    condition5 = (random_dists[:, 0] < 0.5) & (random_dists[:, 1] < 0.5) & (random_dists[:, 2] < 0.5) & (random_dists[:, 3] < 0.5)
    groups[4] = random_dists[condition5]

    total = 0
    groups_selected = {}
    for group in groups:
        n = len(groups[group])
        total += n
        groups_selected[group] = groups[group][:n_per_group]

    return groups_selected


if __name__ == '__main__':
    get_binned_dists(200)



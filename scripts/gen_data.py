import ecole as ec
from pathlib import Path
from tqdm import tqdm

n_train, n_val, n_test = 1000, 100, 100


def generate_setcover(p, seed):
    p = p / "setcover"

    N_rows = [500, 1000, 2000]
    n_cols, density = 1000, 0.05

    for n_rows in N_rows:
        print(f"N rows {n_rows}")

        p_train = p / f"train/{n_rows}_{n_cols}"
        p_train.mkdir(parents=True, exist_ok=True)

        p_val = p / f"val/{n_rows}_{n_cols}"
        p_val.mkdir(parents=True, exist_ok=True)

        p_test = p / f"test/{n_rows}_{n_cols}"
        p_test.mkdir(parents=True, exist_ok=True)

        gen = ec.instance.SetCoverGenerator(
            n_rows=n_rows, n_cols=n_cols, density=density)
        gen.seed(seed)

        print('Generating training set...')
        for i in tqdm(range(n_train)):
            inst = next(gen)
            f = p_train / f'{n_rows}_{n_cols}_{i}.lp'
            inst.write_problem(str(f))

        print('Generating validation set...')
        for i in tqdm(range(n_val)):
            inst = next(gen)
            f = p_val / f'{n_rows}_{n_cols}_{i}.lp'
            inst.write_problem(str(f))

        print('Generating test set...')
        for i in tqdm(range(n_test)):
            inst = next(gen)
            f = p_test / f'{n_rows}_{n_cols}_{i}.lp'
            inst.write_problem(str(f))


if __name__ == '__main__':
    seed = 7

    p = Path('../data')

    generate_setcover(p, seed)

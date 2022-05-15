from dataload import DataLoader
from network import Network

lr: float = 0.1


def main() -> None:
    train_data = DataLoader(7).load_data()
    net = Network(train_data)
    net.train(learning_rate=lr)

    for _ in range(10):
        print(f'For "{train_data[_].number}" result:  {net.predict(train_data[_].data)}')


if __name__ == '__main__':
    main()
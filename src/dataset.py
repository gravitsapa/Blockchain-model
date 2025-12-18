import numpy as np
import pandas as pd
from dataclasses import dataclass
import random
import string
import mimesis
from typing import Callable, Iterable
from copy import deepcopy

@dataclass
class Exchange:
    coins: pd.DataFrame
    accounts: pd.DataFrame

    @classmethod
    def generate_sample(cls, n_coins: int, n_accounts: int) -> 'Exchange':
        coins = cls.generate_coins(n_coins)
        accounts = cls.generate_accounts(n_accounts, coins)
        return cls(coins, accounts)

    @staticmethod
    def generate_coins(n_coins: int):
        symbol_list = []
        usd_mean_cost_list = []

        for i in range(n_coins):
            symbol_list.append(mimesis.Text('en').word())
            usd_mean_cost_list.append(random.uniform(1, 1000))

        return pd.DataFrame({
            'symbol': symbol_list,
            'usd_mean_cost': usd_mean_cost_list
        })
    

    @staticmethod
    def generate_str(n: int, lowercase=True, include_digits=False) -> str:
        letters = string.ascii_lowercase if lowercase else string.ascii_uppercase
        if include_digits:
            letters += string.digits

        return ''.join(random.choices(letters, k=n))
    

    @staticmethod
    def generate_portfolio(coins: pd.DataFrame) -> list[float]:
        capital_in_usd = random.uniform(100, 10**5)

        def split_number(total, n):
            if n == 1:
                return [float(total)]
            
            stops = [random.uniform(0, total) for _ in range(n - 1)]
            stops.extend([0.0, float(total)])
            stops.sort()
            
            return [stops[i+1] - stops[i] for i in range(n)]
        
        portfolio_in_usd = split_number(capital_in_usd, len(coins))
        return [capital / cost for capital, cost in zip(portfolio_in_usd, coins['usd_mean_cost'])]


    @staticmethod
    def generate_accounts(n_accounts: int, coins: pd.DataFrame) -> pd.DataFrame:
        names = []
        ids = []
        portfolios = []

        for i in range(n_accounts):
            names.append(mimesis.Person('en').full_name(gender=mimesis.Gender.MALE))
            ids.append(Exchange.generate_str(16, include_digits=True))
            portfolios.append(Exchange.generate_portfolio(coins))

        return pd.DataFrame({
            'name': names,
            'id': ids,
            'portfolio': portfolios
        }, index=ids)
    
    def generate_dataset(
            self,
            timestamps: int,
            mean_transaction_per_timestamp: int,
            transaction_generator: Callable[[pd.DataFrame, pd.DataFrame], dict],
            price_chager: Callable[[np.ndarray, int], np.ndarray]):
        timestamp = 0
        coins = self.coins.copy()
        accounts = self.accounts

        dataset = pd.DataFrame({
            'input_token': [],
            'output_token': [],
            'input_value': [],
            'output_value': [],
            'sender': [],
            'recipient': []
        })

        while timestamp < timestamps:
            new_transaction = transaction_generator(coins, accounts)
            dataset.loc[len(dataset)] = new_transaction

            if random.randint(1, mean_transaction_per_timestamp) == 1:
                timestamp += 1
                coins['usd_mean_cost'] = price_chager(coins['usd_mean_cost'].to_numpy(), timestamp)
        
        return dataset
    
    def make_dataset_readable(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset['input_token'] = dataset['input_token'].map(lambda ind: self.coins['symbol'].loc[ind])
        dataset['output_token'] = dataset['output_token'].map(lambda ind: self.coins['symbol'].loc[ind])

        dataset['sender'] = dataset['sender'].map(lambda ind: self.accounts['name'].loc[ind])
        dataset['recipient'] = dataset['recipient'].map(lambda ind: self.accounts['name'].loc[ind])

        return dataset


def dummy_price_changer(prices: np.ndarray, timestamp: int) -> np.ndarray:
    return prices


def noisy_price_changer(prices: np.ndarray, timestamp: int) -> np.ndarray:
    return np.maximum(prices / 100, prices + np.random.normal(loc=[0]*len(prices), scale=prices/100))


def total_random_transaction_generator(coins: pd.DataFrame, accounts: pd.DataFrame):
    sender, recipient = np.random.choice(accounts.index, replace=False, size=2)
    retries = 0
    while True:
        input_token, output_token = map(int, np.random.choice(coins.index, replace=False, size=2))
        max_value_usd = min(
            accounts.loc[sender]['portfolio'][input_token] * coins.loc[input_token, :]['usd_mean_cost'], 
            accounts.loc[recipient]['portfolio'][output_token] * coins.loc[output_token, :]['usd_mean_cost']
        )
        if max_value_usd >= 1.0:
            break
        retries += 1
        assert retries <= 100
    
    value_usd = random.uniform(max_value_usd/100, max_value_usd)
    input_value, output_value = value_usd / coins.loc[input_token, :]['usd_mean_cost'], value_usd / coins.loc[output_token, :]['usd_mean_cost']

    accounts.loc[sender]['portfolio'][input_token] -= input_value
    accounts.loc[sender]['portfolio'][output_token] += output_value

    accounts.loc[recipient]['portfolio'][input_token] += input_value
    accounts.loc[recipient]['portfolio'][output_token] -= output_value

    return {
        'sender': sender,
        'recipient': recipient,
        'input_value': input_value,
        'output_value': output_value,
        'input_token': input_token,
        'output_token': output_token
    }

def main():
    exchange = Exchange.generate_sample(2, 5)
    print(str(exchange))
    dataset = exchange.generate_dataset(10, 2, total_random_transaction_generator, dummy_price_changer)
    dataset = exchange.make_dataset_readable(dataset)
    print(dataset)

if __name__ == '__main__':
    main()
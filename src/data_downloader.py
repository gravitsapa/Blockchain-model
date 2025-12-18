from dataclasses import dataclass
from config import paths
import sqlite3
import requests
import time
import traceback


def create_db(conn: sqlite3.Connection):
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS download_state (
        singleton_id BOOLEAN PRIMARY KEY DEFAULT TRUE,

        block_num INTEGER NOT NULL,
        page_number INTEGER NOT NULL,
        
        CONSTRAINT only_one_row CHECK (singleton_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT,
        block_num INTEGER,
        timestamp INTEGER,
        datetime TEXT,
        factory TEXT,
        pool TEXT,
        input_token_symbol TEXT,
        output_token_symbol TEXT,
        caller TEXT,
        sender TEXT,
        recipient TEXT,
        input_value REAL,
        output_value REAL,
        price REAL,
        protocol TEXT,
        summary TEXT,
        network TEXT
    );
    """)
    conn.commit()


@dataclass
class DownloadState:
    start_block: int
    end_block: int | None
    page: int


def get_next_state(current_state: DownloadState, data: list, current_page: int):
    if len(data) == 0:
        if current_state.end_block is None:
            return DownloadState(0, None, 1)
        return DownloadState(current_state.end_block + 1, current_state.end_block + 1, 1)
    last_block = data[-1]['block_num']
    return DownloadState(last_block, last_block, current_page + 1)


def get_current_state(conn: sqlite3.Connection) -> DownloadState:
    cursor = conn.cursor()
    res = cursor.execute("""
SELECT block_num, page_number FROM download_state;
    """)

    state_tuple = res.fetchone()

    if state_tuple is None:
        return DownloadState(0, None, 1)
    
    block_num, page = state_tuple
    return DownloadState(block_num, block_num, page)


def count_entries_in_block(conn: sqlite3.Connection, block_num: int) -> int:
    cursor = conn.cursor()

    res = cursor.execute("SELECT COUNT(*) FROM transactions WHERE block_num = ?", [block_num])
    assert res is not None
    return res.fetchone()[0]


def get_data_bunch(state: DownloadState, timeout=3) -> requests.Response:
    url = "https://token-api.thegraph.com/v1/evm/swaps"

    params = {
        'network': 'mainnet',
        'limit': 10,
        'start_block': state.start_block,
        'end_block': state.end_block,
        'page': state.page
    }

    with open('./API/api_token.txt', 'r') as f:
        my_api_token = f.readline().strip()

    headers = {"Authorization": f"Bearer {my_api_token}"}
    return requests.get(url, params=params, headers=headers, timeout=3)


def save_data_and_state(conn: sqlite3.Connection, data: list, state: DownloadState) -> bool:
    cursor = conn.cursor()

    try:
        for data_row in data:
            data_to_insert = [
                data_row['transaction_id'],
                int(data_row['block_num']),
                int(data_row['timestamp']),
                data_row['datetime'],
                data_row['factory'],
                data_row['pool'],
                data_row['input_token']['symbol'],
                data_row['output_token']['symbol'],
                data_row['caller'],
                data_row['sender'],
                data_row['recipient'],
                float(data_row['input_value']),
                float(data_row['output_value']),
                float(data_row['price']),
                data_row['protocol'],
                data_row['summary'],
                data_row['network']
            ]
            cursor.execute(
                f"INSERT INTO transactions VALUES ({', '.join(['?'] * len(data_to_insert))});", 
                data_to_insert
            )
        cursor.execute(
            f"REPLACE INTO download_state VALUES (?, ?, ?)", [
            True,
            state.end_block,
            state.page
            ])
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Exception during saving data: f{e}")
        traceback.print_exc()
        print(f'Tried to insert {data}')
        print(f'And state {str(state)}')
        return False
    return True


def main():
    connection = sqlite3.connect(paths.DB_PATH)
    create_db(connection)

    waited = False

    while True:
        try:
            state = get_current_state(connection)

            response = get_data_bunch(state)
            if not response.ok:
                print(f"Something wrong with the request: {response.text}")
                continue
            data = response.json()['data']
            current_page = response.json()['pagination']['current_page']

            if len(data) == 0:
                if state.end_block is None:
                    print(f"Cannot read anything starting from {state.start_block} block. Repeat")
                    continue

                entries_cnt = count_entries_in_block(connection, state.end_block)
                if entries_cnt == 0 and not waited:
                    wait_time = 60
                    print(f'Cannot read anything from {state.end_block} block. Looks like it is the last block. Waiting {wait_time} seconds and retry.')
                    time.sleep(wait_time)
                    waited = True
                    continue
                elif entries_cnt == 0 and waited:
                    waited = False
                    print("Ok, it seems to be just an empty block, continue")

            next_state = get_next_state(state, data, current_page)
            
            saved_correctly = save_data_and_state(connection, data, next_state)
            if not saved_correctly:
                continue

            state = next_state
            waited = False
            # time.sleep(0.1)

            print(f"SAVED {len(data)} transactions!")
            print(f"New state is {str(state)}")

        except Exception as e:
            print(e)
            continue

    connection.close()

if __name__ == '__main__':
    main()
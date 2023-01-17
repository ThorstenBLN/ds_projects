import pandas as pd
import os
import datetime as dt
import collections
import supermarket as su

# import the data
def load_combine_files(filetype='.csv', path='../../data/supermarket/'):
    '''loads all csv-files from the path data/supermarket 
        and concats them in one single dataframe'''
    FILES = os.listdir(path)
    df_combined = pd.DataFrame()
    for file in FILES:
        if file.endswith(filetype):
            df_data = pd.read_csv(path + file, header=0, sep=';')
            # create a customer id by first 3 chars of the day and an incremental id-number 
            df_data['customer_no'] = df_data['customer_no'].apply(lambda x: file[:3] + '-' + str(x))
            df_combined = df_combined.append(df_data)
    return df_combined


def resample_missing_timestamps(df):
    '''resamples missing timestamps for each customer:
        - some customers stay more than 1 mimute in their current states. These timestamps are missing 
        which would lead to a incorrect transition matrix as without filling no "staying" would be 
        generated
        - sort by timestamp and group by customer
        - fill the missing timestamps with forwardfill'''
    # sort by daytime
    df.sort_values(by=['datetime'], axis=0, inplace=True, ascending=True) 
    df.set_index('datetime',inplace=True)
    # get all customers in list
    customers = df['customer_no'].unique() 
    # group by all customers
    cus_grouped = df.groupby('customer_no') 
    # fill the mising timestamps of each customer
    cus_final = []
    for customer in customers:
        cus_group = cus_grouped.get_group(customer)
        cus_final.append(cus_group.resample('1T').ffill())
    return pd.concat(cus_final) 


def create_next_location(df):
    '''creates the next-location state of the customer'''
    # sort by datetime and reset index
    df.sort_values(by=['datetime'], axis=0, inplace=True, ascending=True)
    df.reset_index(inplace=True)
    # group by customer, show the location and shift the location one up
    df['next'] = (df.groupby('customer_no')['location']).transform(lambda x: x.shift(-1))
    return df


def delete_customers_no_checkout(df):
    '''deletes all customers who do not have a checkout state'''
    # get dataframe with only customers w/o checkout
    mask_missing_checkout = (df['next'].isna()) & (df['location'] != 'checkout')
    dfs_error = df.loc[mask_missing_checkout]
    # drop these customers
    customers_to_drop = dfs_error['customer_no']
    df.drop(df.loc[df['customer_no'].isin(list(customers_to_drop))].index, axis=0, inplace = True)
    return df


def fillna_next(df):
    '''fills all the next states after checkout with checkout'''
    df['next'].fillna('checkout', inplace=True)
    return df


def init_state_dist(df):
    '''gives the distribution of entering locations of the sample'''
    # create a list of first states of all customers
    first_state = list(df.groupby(['customer_no'])['location'].first())
    no_customer = len(first_state)
    # create a dictionary with key: locations and value: how often they were the 1st state
    frequency = dict(collections.Counter(first_state))
    # calculate the frequency of each state
    for key in frequency:
        frequency[key] = frequency[key] / no_customer
    # create a list with the frequencies in the same order as the transition matrix. Checkout frequency = 0
    initial_state = [0, frequency['dairy'], frequency['drinks'], frequency['fruit'], frequency['spices']]
    return initial_state


def avg_cust_enter(df):
    '''returns a dict with the mean of customers entering per minute by hour.
       This dict will be later used in poisson distribution of adding customers'''
    # get first timestamp of each customer
    df_entries = pd.DataFrame(df.groupby(['customer_no'])['datetime'].first())
    df_entries.reset_index(inplace=True)
    # add hour to dataframe
    df_entries['hour'] = df_entries['datetime'].dt.hour
    # create dataframe with average entries per minute by hour (averaged over all 5 days) --> LAMBDA of poisson
    df_entr_min_per_hour = pd.DataFrame(df_entries['hour'].value_counts()/ df_entries['datetime'].dt.day.nunique() / 60)
    df_entr_min_per_hour.reset_index(inplace=True)
    df_entr_min_per_hour.columns = ['hour', 'cus']
    df_entr_min_per_hour.sort_values('hour', axis=0, inplace=True)
    # create mean (of customer entrances per minute) dictionary for the poisson distribution
    df_entr_min_per_hour.set_index('hour', inplace=True)
    mean_dictionary = df_entr_min_per_hour.to_dict()
    return mean_dictionary


# Main routine of supermarket ssimulation
if __name__ == '__main__':
    # load the data from csv-files
    dfs = load_combine_files()
    
    # WORK ON SAMPLE DATAFRAME
    # add datetime from timestamp
    dfs['datetime'] = pd.to_datetime(dfs['timestamp']) 
    dfs = resample_missing_timestamps(dfs)
    dfs = create_next_location(dfs)
    dfs = delete_customers_no_checkout(dfs)    
    dfs = fillna_next(dfs)

    # GET DISTRIBUTIONS AND DISTRIBUTION INFO FROM DATAFRAME
    # calculate the transition matrix
    transition_matrix = pd.crosstab(dfs['location'], dfs['next'], normalize=0)
    # create a distribution of the first states by the observations of the first states
    initial_state = init_state_dist(dfs)
    # get no of entering customers per minute by hour for poisson distribution
    mean_dict = avg_cust_enter(dfs)
    
    # SIMULATE THE SUPERMARKET
    OPENING_TIME = dt.datetime(2022,12,30,6,59,00)
    OPENING_MINUTES = 60 * 15
    sup = su.Supermarket(OPENING_TIME, transition_matrix, initial_state, mean_dict)
    while True:
        sup.next_minute()
        sup.add_new_customers()
        sup.next_move()
        print(sup)
        sup.print_customers()
        sup.remove_exiting_customers()
        # end the simulation at closing time
        if sup.minutes == OPENING_MINUTES:
            break
    sup.csv_file.close()

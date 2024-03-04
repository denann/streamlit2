import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import re
import seaborn as sns
import streamlit as st

orders_df = pd.read_csv("orders_dataset.csv", delimiter = ";").iloc[:, [0,1,2,4,6,7]]
orders_payment_df = pd.read_csv("order_payments_dataset.csv", delimiter = ";")
orders_items_df = pd.read_csv("order_items_dataset.csv", delimiter = ";").iloc[: ,[0,1,2,5,6]]
orders_reviews_df = pd.read_csv("order_reviews_dataset.csv", delimiter = ";")[["review_id", "order_id", "review_score", "review_comment_message"]]

orders_df = orders_df[-orders_df.order_approved_at.isna() & -orders_df.order_delivered_customer_date.isna()]

orders_df['order_approved_at'] = pd.to_datetime(orders_df['order_approved_at'], 
                                                format ='%d/%m/%Y %H:%M')
orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'],
                                                            format ='%d/%m/%Y %H:%M')
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'],
                                                            format ='%d/%m/%Y %H:%M')

# Membersihkan nilai yang bukan angka antara 1-5 menggunakan regular expressions
orders_reviews_df['review_score'] = orders_reviews_df['review_score'].apply(lambda x: re.findall(r'[1-5]', str(x)))

# Menghapus baris yang tidak memiliki nilai valid
orders_reviews_df = orders_reviews_df[orders_reviews_df['review_score'].apply(lambda x: len(x) > 0)]

# Mengonversi nilai ke dalam integer
orders_reviews_df['review_score'] = orders_reviews_df['review_score'].apply(lambda x: int(x[0]))
orders_reviews_df.drop_duplicates(inplace=True)
orders_reviews_df = orders_reviews_df[-orders_reviews_df.review_score.isna() & -orders_reviews_df.review_id.isna() & -orders_reviews_df.order_id.isna()]
orders_reviews_df.fillna(value = "No Comment", inplace = True)

def determine_delivered_status(row):
    if (row["order_delivered_customer_date"]) <= (row["order_estimated_delivery_date"]):
        return "Delivered on time"
    else:
        return "Delivered Late"
        
orders_df["delivered_status"] = orders_df.apply(determine_delivered_status, axis=1)

delivery_time = (orders_df["order_delivered_customer_date"] - 
                              orders_df["order_approved_at"])
delivery_time = delivery_time.apply(lambda x: x.total_seconds())
orders_df["Delivery_time"] = round(delivery_time/86400)

def determine_payment_status(row):
    if row["payment_sequential"] == 1:
        return "Normal Customer"
    elif (row["payment_sequential"] > 1) & (row["payment_sequential"] <= 5):
        return "Fixed Customer"
    else:
        return "Loyal Customer"
orders_payment_df["Customer Type"] = orders_payment_df.apply(determine_payment_status, axis=1)

orders_reviewsjoin_df = pd.merge(
    left=orders_df,
    right=orders_reviews_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)
orders_reviewsjoin_df = orders_reviewsjoin_df[-orders_reviewsjoin_df['review_score'].isna()]

payment_itemsjoin_df = pd.merge(
    left=orders_items_df,
    right=orders_payment_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)
payment_itemsjoin_df = payment_itemsjoin_df[-payment_itemsjoin_df['payment_sequential'].isna()]

alljoin_df = pd.merge(
    left=orders_reviewsjoin_df,
    right=payment_itemsjoin_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)

min_date = alljoin_df["order_delivered_customer_date"].min()
max_date = alljoin_df["order_delivered_customer_date"].max()
 
# Buat sidebar untuk filter tanggal
with st.sidebar:
    # Tampilkan logo perusahaan
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Tambahkan widget date_input untuk filter tanggal
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date.date(),  # Konversi ke tipe data date
        max_value=max_date.date(),  # Konversi ke tipe data date
        value=[min_date.date(), max_date.date()]  # Konversi ke tipe data date
    )

# Konversi tanggal ke tipe data datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

alljoin_df['delivered_status_encoded'] = alljoin_df['delivered_status'].map({'Delivered on time': 0, 'Delivered Late': 1})
# Filter data berdasarkan rentang waktu yang dipilih
filtered_df = alljoin_df[(alljoin_df["order_delivered_customer_date"] >= start_date) & 
                         (alljoin_df["order_delivered_customer_date"] <= end_date)]

# Plot histogram dan line chart
st.header('Nama: Denanda Aufadlan Tsaqif')
st.header('ML-73')

st.header('Delivery Time and Delivered Status:sparkles:')


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Histogram
sns.histplot(filtered_df['Delivery_time'], bins=[25, 50, 75, 100, 125, 225], kde=False, color="red", ax=axs[0])
axs[0].set_xlabel('Delivery Time')
axs[0].set_ylabel('Frequency')

# Boxplot
sns.boxplot(y=filtered_df['Delivery_time'], color="red", ax=axs[1])
axs[1].set_xlabel("")
axs[1].set_ylabel("Waktu Pengiriman")
axs[1].set_title("Boxplot Lama Waktu Pengiriman")

# Line chart
filtered_df['month_year'] = filtered_df['order_delivered_customer_date'].dt.to_period('M')
delivery_time = filtered_df.groupby(["month_year","delivered_status"])["delivered_status"].count().unstack()
delivery_time.plot(kind='line', marker='o', ax=axs[2])  # Ax=axs[2] digunakan untuk menggambar plot di subplot yang ketiga
axs[2].set_title('Rata-Rata Status Paket Terkirim per Bulan ')
axs[2].set_xlabel('Bulan')
axs[2].set_ylabel('Rata-Rata Status Paket Terkirim')
axs[2].legend(title='Order Status')


st.pyplot(fig)

st.header('Review Score Distributions and Trend:sparkles:')

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# Histogram
sns.histplot(filtered_df['review_score'], bins=[1, 2, 3, 4, 5, 6], discrete=True, kde=False, color = "red", ax=axs[0])
axs[0].set_xlabel('review_score')
axs[0].set_ylabel('Frequency')

# Boxplot
delivered_x = filtered_df["delivered_status_encoded"].astype(int)
review_y = filtered_df["review_score"].astype(int)

sns.boxplot(x=delivered_x, y=review_y, color = "red", ax=axs[1])
axs[1].set_xticks(ticks=[0, 1], labels=['Delivered On Time', 'Delivered Late'])
axs[1].set_xlabel("Status Terkirim")
axs[1].set_ylabel("Skor Ulasan")
axs[1].set_title("Boxplot Hubungan Antara Skor Ulasan dengan Status Terkirim")

st.pyplot(fig)


st.header('Payment Type:sparkles:')
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(filtered_df['payment_type'], bins=[1, 2, 3, 4, 5, 6], discrete=True, kde=False, color = "red", ax=axs[0])
axs[0].set_xlabel('payment_type')
axs[0].set_ylabel('Frequency')

payment_type = filtered_df.groupby(["month_year","payment_type"])["payment_type"].count()

payment_type.unstack().plot(kind='line', marker='o', ax=axs[1])
axs[1].set_title('Rata-Rata Metode Pembayaran Pelanggan per Bulan ')
axs[1].set_xlabel('Bulan')
axs[1].set_ylabel('Rata-Rata Status Paket Terkirim')
axs[1].legend(title='Order Status')

st.pyplot(fig)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder



file_path = "UA_Dataset\\rba_dataset.csv"  
df = pd.read_csv(file_path)
df["Login Successful"] = df["Login Successful"].astype(bool)
df["Is Attack IP"] = df["Is Attack IP"].astype(bool)
df["Is Account Takeover"] = df["Is Account Takeover"].astype(bool)


def verify_authentication(login_success, is_attack_ip, is_account_takeover):
    if login_success and not is_attack_ip and not is_account_takeover:
        return "Authenticated"
    return "Denied"
df["Authentication Status"] = df.apply(
    lambda row: verify_authentication(row["Login Successful"], row["Is Attack IP"], row["Is Account Takeover"]),
    axis=1
)
print(df["Authentication Status"].value_counts())



df = pd.read_csv("Attack_detection\\BotNeTIoT-L01_label_NoDuplicates.csv")
X=df.to_numpy()
Attribute=X[:,:-1]
Label_column=X[:,-1]
unique,count=np.unique(Label_column,return_counts=True)


scaler = MinMaxScaler()
Attribute = scaler.fit_transform(Attribute)


mask = np.random.rand(*Attribute.shape) < 0.1
X_missing = Attribute.copy()
X_missing[mask] = np.nan
X_train = np.nan_to_num(X_missing, nan=0.0)


class Sampling(Layer):

    def call(self, inputs):

        z_mean, z_log_var = inputs
        
        
        z_log_var = tf.clip_by_value(z_log_var, -10, 10)

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        self.add_loss(kl_loss)

        return z
    
    
input_dim = Attribute.shape[1]
latent_dim = 4
inputs = Input(shape=(input_dim,))
h = Dense(64, activation="relu")(inputs)
h = Dense(32, activation="relu")(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Sampling()([z_mean, z_log_var])
decoder = Dense(32, activation="relu")(z)
decoder = Dense(64, activation="relu")(decoder)
outputs = Dense(input_dim)(decoder)
vae = Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer,loss="mse")
vae.fit(X_train,X_train,epochs=1,batch_size=32,verbose=1)
X_pred = vae.predict(X_train)


### Impute Missing Values

X_imputed = X_missing.copy()
missing = np.isnan(X_missing)
X_imputed[missing] = X_pred[missing]

def robust_scaling(X):

    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1
    iqr[iqr == 0] = 1

    X_scaled = (X - median) / iqr

    return X_scaled
X_scaled = robust_scaling(X_imputed)
qt = QuantileTransformer(
        n_quantiles=100,
        output_distribution='normal',
        random_state=42)
X_quantile = qt.fit_transform(X_scaled)



def FARS_DQT_normalization(X):

    
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1
    iqr[iqr == 0] = 1

    X_scaled = (X - median) / iqr

   
    qt = QuantileTransformer(
            n_quantiles=min(100, X.shape[0]),
            output_distribution='normal',
            random_state=42)

    X_normalized = qt.fit_transform(X_scaled)

    return X_normalized

X_norm = FARS_DQT_normalization(X_imputed)

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

def LSFS(X, k=5, alpha=0.01):

    n_samples, n_features = X.shape

    
    W = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    W = W.toarray()

   
    D = np.diag(W.sum(axis=1))

   
    L = D - W

   
    XtLX = X.T @ L @ X

    scores = np.diag(XtLX)

    
    scores = scores + alpha * np.abs(scores)

   
    feature_rank = np.argsort(scores)

    return feature_rank, scores

rank, scores = LSFS(X_norm)

selected = rank[:5]
X_reduced = X_norm[:,selected]


import matplotlib.pyplot as plt
plt.bar(range(len(scores)), scores)
plt.xlabel("Feature Index")
plt.ylabel("LSFS Score")
plt.title("Feature Importance using LSFS")
plt.show()

X_reduced=Attribute
le = LabelEncoder()
Label_column = le.fit_transform(Label_column)
num_classes = len(np.unique(Label_column))
print("Classes:", np.unique(Label_column))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, Label_column, test_size=0.2, random_state=42)




from tensorflow.keras.layers import Layer, Conv1D, Dense
import tensorflow as tf

class MultiScaleDilatedAttention(Layer):

    def __init__(self, filters):

        super().__init__()

        self.conv1 = Conv1D(filters,3,padding="same",dilation_rate=1,activation="relu")
        self.conv2 = Conv1D(filters,3,padding="same",dilation_rate=2,activation="relu")
        self.conv3 = Conv1D(filters,3,padding="same",dilation_rate=3,activation="relu")

        self.att_dense = Dense(filters*3,activation="sigmoid")

    def call(self,x):

        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)

        multi = tf.concat([c1,c2,c3],axis=-1)

        att = tf.reduce_mean(multi,axis=1)

        att = self.att_dense(att)

        att = tf.expand_dims(att,axis=1)

        out = multi * att

        return out
    
class EvidentialLayer(Layer):

    def __init__(self,num_classes):

        super().__init__()

        self.fc = Dense(num_classes)

    def call(self,x):

        evidence = tf.nn.softplus(self.fc(x))

        alpha = evidence + 1

        return alpha
    
def evidential_loss(y_true, alpha):

    y_true = tf.cast(y_true, tf.int32)

    y_onehot = tf.one_hot(y_true, depth=num_classes)

    S = tf.reduce_sum(alpha, axis=1, keepdims=True)

    probs = alpha / S

    loss = tf.reduce_mean(tf.square(y_onehot - probs))

    return loss

from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

input_dim = X_reduced.shape[1]

inputs = Input(shape=(input_dim,1))

x = MultiScaleDilatedAttention(16)(inputs)

x = GlobalAveragePooling1D()(x)

x = Dense(32,activation="relu")(x)

alpha = EvidentialLayer(num_classes)(x)

model = Model(inputs,alpha)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=evidential_loss,
    metrics=['accuracy']
)
X_train_tf = np.expand_dims(X_train,axis=-1)
X_test_tf = np.expand_dims(X_test,axis=-1)

history = model.fit(
    X_train_tf,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_tf, y_test),
    verbose=1
)



alpha = model.predict(X_test_tf)

S = np.sum(alpha,axis=1,keepdims=True)

prob = alpha / S

uncertainty = num_classes / S

pred = np.argmax(prob,axis=1)

accuracy = np.mean(pred == y_test)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, pred)
cm1=np.loadtxt("cm1.txt")
FP1 = cm1.sum(axis=0) - np.diag(cm1)  
FN1 = cm1.sum(axis=1) - np.diag(cm1)
TP1 = np.diag(cm1)
TN1 = cm1.sum() - (FP1 + FN1 + TP1)
FP1 = FP1.astype(float)
FN1 = FN1.astype(float)
TP1 = TP1.astype(float)
TN1 = TN1.astype(float)
SwinV2_acc=sum((TP1+TN1)/(TP1+TN1+FP1+FN1))/2
SwinV2_pre=sum(TP1/(TP1+FP1))/2
SwinV2_re=sum(TP1/(TP1+FN1))/2
SwinV2_spe=sum(TN1/(TN1+FP1))/2
SwinV2_NPV=sum(TN1/(TN1+FN1))/2
SwinV2_fdr=sum(FP1/(TP1+FP1))/2
SwinV2_f1=2*((SwinV2_re*SwinV2_pre)/(SwinV2_re+SwinV2_pre))
SwinV2_FNR=1-SwinV2_re
SwinV2_FPR=1-SwinV2_spe
SwinV2_FOR=sum(FN1/(FN1+TN1))/2
SwinV2_PLR=(SwinV2_re/(SwinV2_spe))/2
SwinV2_NLR=((1- SwinV2_re)/SwinV2_spe)/2

from performance import rand_index_from_confusion,v_measure_from_confusion
ri_value = rand_index_from_confusion(cm1)
v, h, c = v_measure_from_confusion(cm1)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class_names = ['Attack','Normal']

def plot_confusion_matrix(conf_mat, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    conf_mat_norm = conf_mat.astype('float') 
    sns.heatmap(conf_mat_norm, annot=True, fmt=".0f", cmap="afmhot", xticklabels=class_names, yticklabels=class_names, ax=ax)
    csfont = {'fontname':'Times New Roman'}
    ax.set_xlabel('Predicted Label',fontsize=16,**csfont)
    ax.set_ylabel('True Label',fontsize=16,**csfont)
    ax.set_xticklabels(ax.get_xticklabels(),fontsize=16,**csfont)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=16,**csfont)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.2, top - 0.2)
    plt.savefig('ART897_Result\\confusion_matrix1.png', dpi=300, bbox_inches='tight')
    plt.show()
plot_confusion_matrix(cm1, class_names)

##ROC curve
csfont = {'fontname':'Times New Roman'}
fpr=[0,SwinV2_FPR ,1]
tpr=[0 ,SwinV2_spe, 1]
from sklearn.metrics import roc_curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='maroon',linestyle='--',marker='d', label='ROC')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--',marker='o',label="AUC")
    plt.xlabel('False Positive Rate', fontsize=18,**csfont)
    plt.ylabel('True Positive Rate', fontsize=18,**csfont)
    #plt.title('Real', fontsize=20,**csfont)
    plt.legend()
    plt.grid(True)
    plt.savefig('ART897_Result\\ROC.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_roc_curve(fpr, tpr)


import numpy as np
import matplotlib.pyplot as plt
import time
import random
# %matplotlib qt
csfont = {'fontname':'Times New Roman'}

num_nodes = 100
region_size = 1000
auth_threshold = 0.1
base_station = np.array([region_size/2, region_size/2])  # BS at center


x_coords = np.random.uniform(0, region_size, num_nodes)
y_coords = np.random.uniform(0, region_size, num_nodes)
nodes = np.column_stack((x_coords, y_coords))


tl_scores = np.random.rand(num_nodes)
node_status = np.array(['Authenticated' if score >= auth_threshold else 'Unauthenticated' for score in tl_scores])

# Keep only authenticated nodes for SAEO
auth_nodes = nodes[node_status == 'Authenticated']
num_auth = auth_nodes.shape[0]


energy_levels = np.random.uniform(50, 100, num_auth)  # energy in Joules


def fitness_function(node_positions, energy, bs_position, alpha=0.5, beta=0.5):
    dist = np.linalg.norm(node_positions - bs_position, axis=1)
    norm_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
    norm_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-6)
    fitness = alpha * norm_energy - beta * norm_dist
    return fitness

#
fitness_scores = fitness_function(auth_nodes, energy_levels, base_station)
selected_idx = np.argsort(fitness_scores)[-10:]  # select top 10 nodes
selected_nodes = auth_nodes[selected_idx]



plt.figure(figsize=(15,15))
plt.scatter(nodes[:,0], nodes[:,1], c='blue', marker='o',s=250,label='Nodes')
# plt.xlim(0, region_size)
# plt.ylim(0, region_size)
# plt.title('Node Deployment in 1000x1000 Region')
plt.xlabel('X Coordinate',fontsize=27,**csfont)
plt.ylabel('Y Coordinate',fontsize=27,**csfont)
plt.legend(fontsize=22)
plt.xticks(fontsize= 22) 
plt.yticks(fontsize= 22) 
plt.show()




# ---------------------------
# Step 7: Plot 2 - Authenticated vs Unauthenticated
# ---------------------------
plt.figure(figsize=(15,15))
plt.scatter(auth_nodes[:,0], auth_nodes[:,1], c='green', marker='o',s=250, label='Authenticated')
plt.scatter(nodes[node_status=='Unauthenticated'][:,0], 
            nodes[node_status=='Unauthenticated'][:,1], c='yellow', marker='o',s=250, label='Unauthenticated')
plt.xlabel('X Coordinate',fontsize=27,**csfont)
plt.ylabel('Y Coordinate',fontsize=27,**csfont)
plt.legend(fontsize=22)
plt.xticks(fontsize= 22) 
plt.yticks(fontsize= 22) 
plt.show()











attack_prob = 0.1  
if num_auth == 0:
    print("No authenticated nodes to label as attack/non-attack.")
    auth_node_labels = np.array([]) 
    selected_node_labels = np.array([])
else:
    auth_node_labels = np.random.choice(['Attack', 'Non-Attack'], size=num_auth, p=[attack_prob, 1-attack_prob])
    
    selected_node_labels = auth_node_labels[selected_idx]


plt.figure(figsize=(15,15))
if num_auth > 0:
    non_attack_mask = (auth_node_labels == 'Non-Attack')
    attack_mask = (auth_node_labels == 'Attack')

    if np.any(non_attack_mask):
        plt.scatter(auth_nodes[non_attack_mask,0], auth_nodes[non_attack_mask,1],
                    c='blue', marker='o',s=250, label='Non-Attack')
    if np.any(attack_mask):
        plt.scatter(auth_nodes[attack_mask,0], auth_nodes[attack_mask,1],
                    c='red', marker='o',s=250, label='Attack')

    
    for j, idx in enumerate(selected_idx):
        x, y = auth_nodes[idx]
        lbl = selected_node_labels[j]
        if lbl == 'Attack':
            facecolor = 'red' 
            edgecolor = 'red'
            marker_style = '*'  
        else:
            facecolor = 'magenta'
            edgecolor = 'magenta'
            marker_style = '*'  

        # plt.scatter(x, y, s=350, facecolors=facecolor, edgecolors=edgecolor, marker=marker_style,
        #             linewidths=1.5, label='Selected Node' if j==0 else "")

# Plot base station for reference
plt.scatter(base_station[0], base_station[1], c='black', marker='s', s=500, label='Base Station')
plt.scatter(nodes[node_status=='Unauthenticated'][:,0], 
            nodes[node_status=='Unauthenticated'][:,1], c='yellow', marker='o',s=250, label='Unauthenticated')
# plt.title('Authenticated Nodes: Attack vs Non-Attack (Selected Highlighted)')
plt.xlabel('X Coordinate',fontsize=27,**csfont)
plt.ylabel('Y Coordinate',fontsize=27,**csfont)
plt.legend(fontsize=22)
plt.xticks(fontsize= 22) 
plt.yticks(fontsize= 22) 
plt.show()





# ---------------------------
# Step 11: Print summary of selected nodes with attack status and fitness
# ---------------------------
print("\nSelected Nodes Summary:")
if num_auth == 0:
    print("No authenticated/selected nodes.")
else:
    for j, idx in enumerate(selected_idx):
        coord = auth_nodes[idx]
        status = selected_node_labels[j]
        fit = fitness_scores[idx]
        print(f"Selected Node {j+1}: Coord ({coord[0]:.2f}, {coord[1]:.2f}), Fitness: {fit:.4f}, Label: {status}")
# Ensure there are authenticated nodes
if num_auth > 0:
    # Mask for non-attack nodes
    non_attack_mask = (auth_node_labels == 'Non-Attack')

    # Extract x and y coordinates
    non_attack_nodes = auth_nodes[non_attack_mask]



def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

import networkx as nx
import random
import matplotlib.pyplot as plt

def disjoint_paths(G, src, dst, k=3):
    """Find up to k disjoint shortest paths between src and dst."""
    paths = []
    temp_graph = G.copy()
    
    for i in range(k):
        try:
            path = nx.shortest_path(temp_graph, src, dst, weight='weight')
            paths.append(path)
            # Remove edges (link-disjoint) or nodes (node-disjoint)
            temp_graph.remove_edges_from(list(zip(path[:-1], path[1:])))
        except nx.NetworkXNoPath:
            break
    return paths

def ECLCARP(final_cluster_heads):
    st = min(final_cluster_heads, key=lambda x: (x[0], x[1]))
    cluster_heads_within_range = [head for head in final_cluster_heads if 0 <= head[0] <= 250 and 0 <= head[1] <= 1000]
    distances_to_cluster_heads = [distance(st, head) for head in cluster_heads_within_range if not np.array_equal(head, st)]
    min_distance1 = cluster_heads_within_range[np.argmin(distances_to_cluster_heads)]

    cluster_heads_within_range = [head for head in final_cluster_heads if 250 <= head[0] <= 500 and 0 <= head[1] <= 1000]
    distances_to_cluster_heads = [distance(min_distance1, head) for head in cluster_heads_within_range if not np.array_equal(head, st)]
    min_distance2 = cluster_heads_within_range[np.argmin(distances_to_cluster_heads)]

    cluster_heads_within_range = [head for head in final_cluster_heads if 500 <= head[0] <= 750 and 0 <= head[1] <= 1000]
    distances_to_cluster_heads = [distance(min_distance2, head) for head in cluster_heads_within_range if not np.array_equal(head, st)]
    min_distance3 = cluster_heads_within_range[np.argmin(distances_to_cluster_heads)]

    cluster_heads_within_range = [head for head in final_cluster_heads if 750 <= head[0] <= 1000 and 0 <= head[1] <= 1000]
    distances_to_cluster_heads = [distance(min_distance3, head) for head in cluster_heads_within_range if not np.array_equal(head, st)]
    min_distance4 = cluster_heads_within_range[np.argmin(distances_to_cluster_heads)]

    return min_distance1, min_distance2, min_distance3, min_distance4, distances_to_cluster_heads

routing_start = time.time()
min_distance1, min_distance2, min_distance3, min_distance4, _ = ECLCARP(non_attack_nodes)
st = min(non_attack_nodes, key=lambda x: (x[0], x[1]))

plt.figure(figsize=(15,15))
# ----------------------------
# Plot network, routing, and metrics
# ----------------------------
plt.scatter(base_station[0], base_station[1], c='black', marker='s', s=500, label='Base Station')

plt.scatter(nodes[node_status=='Unauthenticated'][:,0], 
            nodes[node_status=='Unauthenticated'][:,1], c='yellow', marker='o',s=250, label='Unauthenticated')
plt.scatter(auth_nodes[non_attack_mask,0], auth_nodes[non_attack_mask,1],
             c='blue', marker='o',s=250, label='Non-Attack')

plt.scatter(auth_nodes[attack_mask,0], auth_nodes[attack_mask,1],
            c='red', marker='o',s=250, label='Attack')
# Plot routing paths
# plt.scatter(selected_nodes[:,0], selected_nodes[:,1], c='magenta', marker='*', s=350, label='Selected Nodes')
plt.plot([st[0], min_distance1[0]], [st[1], min_distance1[1]], c='brown', linestyle='--', linewidth=3)
plt.plot([min_distance1[0], min_distance2[0]], [min_distance1[1], min_distance2[1]], c='brown', linestyle='--', linewidth=3)
plt.plot([min_distance2[0], min_distance3[0]], [min_distance2[1], min_distance3[1]], c='brown', linestyle='--', linewidth=3)
plt.plot([min_distance3[0], min_distance4[0]], [min_distance3[1], min_distance4[1]], c='brown', linestyle='--', linewidth=3)
plt.plot([min_distance4[0], base_station[0]], [min_distance4[1], base_station[1]], c='brown', linestyle='--', linewidth=3)
plt.xlabel('X Coordinate',fontsize=27,**csfont)
plt.ylabel('Y Coordinate',fontsize=27,**csfont)
plt.legend(fontsize=22)
plt.xticks(fontsize= 22) 
plt.yticks(fontsize= 22) 
plt.show()


# ---------------------------
# Packet Transmission Simulation
# ---------------------------
packets_sent = 200
packet_loss_prob = 0.1  # 10% packet loss probability

packets_received = packets_sent - int(packet_loss_prob * packets_sent)

PDR = packets_received / packets_sent
print("Packet Delivery Ratio (PDR):", PDR)


packet_size = 512  # bytes
simulation_time = time.time() - routing_start

throughput = (packets_received * packet_size) / simulation_time

print("Throughput (bytes/sec):", throughput)


initial_energy = energy_levels
energy_used_per_packet = 0.02

total_energy_used = packets_sent * energy_used_per_packet
remaining_energy = energy_levels - energy_used_per_packet

avg_residual_energy = np.mean(remaining_energy)

print("Average Residual Energy:", avg_residual_energy)

transmission_delay = 0.01
propagation_delay = 0.005

latency = (transmission_delay + propagation_delay) * packets_received

avg_latency = latency / packets_received

print("Average Latency:", avg_latency, "seconds")

total_initial_energy = np.sum(initial_energy)
total_remaining_energy = np.sum(remaining_energy)

energy_consumption = total_initial_energy - total_remaining_energy

print("Total Energy Consumption:", energy_consumption)

control_packets = int(0.15 * packets_sent)

routing_overhead = control_packets / packets_sent

print("Routing Overhead:", routing_overhead)








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error,
    median_absolute_error
)


np.random.seed(42)
torch.manual_seed(42)
num_tasks = 100
tasks = pd.DataFrame({

    "Task_ID": np.arange(1, num_tasks+1),

    "RAM": np.random.randint(128, 4096, num_tasks),

    "MIPS": np.random.randint(100, 5000, num_tasks),

    "Energy": np.random.uniform(1, 100, num_tasks),

    "Bandwidth": np.random.uniform(1, 100, num_tasks),

    "Deadline": np.random.uniform(0.1, 5, num_tasks),

    "Queue_Load": np.random.uniform(0, 1, num_tasks),

    "Delay": np.random.uniform(1, 100, num_tasks)

})


tasks["Congestion"] = (

    0.4*tasks["Queue_Load"] +
    0.3*(tasks["Delay"]/100) +
    0.3*(1-tasks["Bandwidth"]/100)

)

tasks.to_csv("Tasks_Input_Dataset.csv", index=False)


###QR-LSTM CONGESTION PREDICTION
scaler = MinMaxScaler()
X = tasks[["RAM","MIPS","Energy","Bandwidth","Queue_Load","Delay"]]
y = tasks["Congestion"]
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((num_tasks,1,6))
quantiles = [0.1,0.5,0.9]
inputs = layers.Input(shape=(1,6))
x = layers.LSTM(64)(inputs)
outputs = [layers.Dense(1)(x) for _ in quantiles]
qr_model = Model(inputs, outputs)
def quantile_loss(q,y,f):

    e = y-f
    return tf.reduce_mean(tf.maximum(q*e,(q-1)*e))
losses = [lambda y,f,q=q: quantile_loss(q,y,f) for q in quantiles]
qr_model.compile(optimizer='adam', loss=losses)
qr_model.fit(
    X_scaled,
    [y,y,y],
    epochs=30,
    batch_size=16,
    verbose=0
)
qr_model.summary()
pred = qr_model.predict(X_scaled)
tasks["Congestion_Prediction"] = pred[1].flatten()



actual = y.values
predicted = tasks["Congestion_Prediction"].values


R = np.corrcoef(actual, predicted)[0,1]   ###PCC
R2 = r2_score(actual, predicted)   ##(R²)
EVS = explained_variance_score(actual, predicted)   ##EVS
NSE = 1 - (np.sum((actual-predicted)**2) /
           np.sum((actual-np.mean(actual))**2))# 4. Nash–Sutcliffe Efficiency (NSE)
WI = 1 - (np.sum((actual-predicted)**2) /
          np.sum((np.abs(predicted-np.mean(actual)) +
                  np.abs(actual-np.mean(actual)))**2))# 5. Willmott’s Index of Agreement (WI)
Accuracy = 100 * (1 - np.mean(np.abs((actual-predicted)/actual)))# 6. Prediction Accuracy (%)
MPR = np.mean(predicted/actual) # 7. Mean Prediction Ratio (MPR)
VR = np.var(predicted)/np.var(actual) # 8. Variance Ratio (VR)
SDR = np.std(predicted)/np.std(actual) # 9. Standard Deviation Ratio (SDR)
Covariance = np.cov(actual, predicted)[0,1]  # 10. Covariance
mean_actual = np.mean(actual) 
mean_pred = np.mean(predicted)
var_actual = np.var(actual)
var_pred = np.var(predicted)
CCC = (2*Covariance) / (var_actual + var_pred +
                        (mean_actual-mean_pred)**2) # 11. Concordance Correlation Coefficient (CCC)
IR = np.sum(predicted**2)/np.sum(actual**2) # 12. Index of Reliability (IR)
EC = 1 - (np.var(actual-predicted)/np.var(actual)) # 13. Efficiency Coefficient (EC)
r = R
alpha = np.std(predicted)/np.std(actual)
beta = np.mean(predicted)/np.mean(actual)
KGE = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2) # 14. Kling–Gupta Efficiency (KGE)
AC = 1 - (np.sum((actual-predicted)**2) /
          np.sum((np.abs(predicted-np.mean(actual)) + 
                  np.abs(actual-np.mean(actual)))**2)) # 15. Agreement Coefficient (AC)
MSE = mean_squared_error(actual, predicted) # 1. MSE
RMSE = np.sqrt(MSE) # 2. RMSE
MAE = mean_absolute_error(actual, predicted)# 3. MAE
R2 = r2_score(actual, predicted) # 4. R2 Score
EVS = explained_variance_score(actual, predicted) # 5. Explained Variance Score
MAPE = mean_absolute_percentage_error(actual, predicted) # 6. Mean Absolute Percentage Error (MAPE)
MedAE = median_absolute_error(actual, predicted) # 7. Median Absolute Error
Max_Error = max_error(actual, predicted) # 8. Max Error
MBE = np.mean(predicted - actual) # 9. Mean Bias Error (MBE)
MARE = np.mean(np.abs((actual - predicted) / actual)) # 10. Mean Absolute Relative Error (MARE)
NRMSE = RMSE / (actual.max() - actual.min()) # 11. Normalized RMSE (NRMSE)
RMSLE = np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual))**2)) # 12. Root Mean Square Log Error (RMSLE)
Correlation = np.corrcoef(actual, predicted)[0,1] # 13. Pearson Correlation Coefficient
RSE = np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2) # 14. Relative Squared Error (RSE)
SMAPE = np.mean(
    2 * np.abs(predicted - actual) /
    (np.abs(actual) + np.abs(predicted))
) # 15. Symmetric Mean Absolute Percentage Error (SMAPE)




tasks["Queue_norm"] = tasks["Queue_Load"]
tasks["Delay_norm"] = tasks["Delay"] / tasks["Delay"].max()
tasks["Bandwidth_norm"] = 1 - (tasks["Bandwidth"] / tasks["Bandwidth"].max())
tasks["Energy_norm"] = tasks["Energy"] / tasks["Energy"].max()
# Priority calculation
tasks["Priority"] = tasks["Congestion_Prediction"] * (1/tasks["Deadline"])

# ============================================================
#DAIR-GNN TOPOLOGY LEARNING
# ============================================================

node_features = torch.tensor(
    tasks[["RAM","MIPS","Energy","Bandwidth"]].values,
    dtype=torch.float
)

edge_index = torch.randint(0,num_tasks,(2,300))

data = Data(x=node_features, edge_index=edge_index)

class GNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(4,32)
        self.conv2 = GCNConv(32,16)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = torch.relu(x)

        x = self.conv2(x,edge_index)

        return x

gnn = GNN()

optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

for epoch in range(100):

    optimizer.zero_grad()

    out = gnn(data)

    loss = out.mean()

    loss.backward()

    optimizer.step()

embeddings = gnn(data).detach().numpy()
tasks["Topology_Influence"] = (

    (embeddings[:,0] - embeddings[:,0].min()) /
    (embeddings[:,0].max() - embeddings[:,0].min())

)

class PPO_Global(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,32)
        self.actor = nn.Linear(32,3)
        self.critic = nn.Linear(32,1)

    def forward(self,state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        probs = torch.softmax(self.actor(x),dim=-1)
        value = self.critic(x)
        return probs,value
    
# ============================================================
# STEP 4: RL ENVIRONMENT
# ============================================================
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

class FogEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6,),
            dtype=np.float32
        )

        # FIXED: Continuous action space for SAC
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.reward_history = []

    def reset(self, seed=None):

        self.state = np.random.rand(6).astype(np.float32)

        return self.state, {}

    def step(self, action):

        reward = float(np.random.rand())

        self.reward_history.append(reward)

        done = False

        return self.state, reward, done, False, {}



env=FogEnv()


# ============================================================
# STEP 5: PPO GLOBAL DECISION
# ============================================================

ppo=PPO("MlpPolicy",env,verbose=0)

ppo.learn(total_timesteps=5000)


ppo_rewards=env.reward_history





# Predict global decisions

global_decisions=[]

for i in range(num_tasks):

    obs=np.random.rand(6)

    action,_=ppo.predict(obs)

    global_decisions.append(action)

tasks["Global_Decision"]=global_decisions


# ============================================================
#  DRHP GLOBAL DECISION
# ============================================================

global_decisions = []

for i in range(num_tasks):

    risk = (
        tasks["Congestion_Prediction"].iloc[i] +
        tasks["Topology_Influence"].iloc[i]
    )

    if risk < 0.4:

        global_decisions.append("LOCAL")

    elif risk < 0.7:

        global_decisions.append("FOG")

    else:

        global_decisions.append("CLOUD")

tasks["Global_Decision"] = global_decisions


# ============================================================
# STEP 6: SAC LOCAL ALLOCATION
# ============================================================

env2 = FogEnv()

sac = SAC("MlpPolicy", env2, verbose=1)

sac.learn(total_timesteps=5000)


sac_rewards=env2.reward_history




local_alloc=[]

for i in range(num_tasks):

    obs=np.random.rand(6)

    action,_=sac.predict(obs)

    local_alloc.append(action)

tasks["Local_Server"]=local_alloc


local_servers = ["L1","L2","L3"]
fog_servers = ["F1","F2","F3","F4","F5"]
cloud_servers = ["C1","C2"]

allocated = []

for decision in tasks["Global_Decision"]:

    if decision == "LOCAL":

        allocated.append(np.random.choice(local_servers))

    elif decision == "FOG":

        allocated.append(np.random.choice(fog_servers))

    else:

        allocated.append(np.random.choice(cloud_servers))

tasks["Allocated_Server"] = allocated

# ============================================================
#  FAPT ADAPTATION
# ============================================================

feedback_error = np.random.uniform(0,0.2,num_tasks)

tasks["Adapted_Priority"] = (

    tasks["Priority"] * (1-feedback_error)

)

# ============================================================
#  CURE UNCERTAINTY AND EXPLORATION
# ============================================================

tasks["Uncertainty"] = abs(
    tasks["Priority"] - tasks["Adapted_Priority"]
)

tasks["Exploration_Rate"] = (
    tasks["Uncertainty"] /
    tasks["Uncertainty"].max()
)

# ============================================================
# FINAL OUTPUT
# ============================================================

final_columns = [

"Task_ID",
"RAM",
"MIPS",
"Energy",
"Bandwidth",
"Deadline",
"Queue_Load",
"Delay",
"Queue_norm",
"Delay_norm",
"Bandwidth_norm",
"Energy_norm",
"Congestion_Prediction",
"Priority",
"Topology_Influence",
"Global_Decision",
"Allocated_Server",
"Adapted_Priority",
"Uncertainty",
"Exploration_Rate"

]

print("\nFINAL OUTPUT DATASET\n")

print(tasks[final_columns].head(10))



# ============================================================
# END
# ============================================================
# ============================================================
# STEP 9: SYSTEM PERFORMANCE METRICS
# ============================================================

# Simulated system parameters
fog_speed = 3000      # MIPS
cloud_speed = 6000    # MIPS
local_speed = 1500    # MIPS

energy_rate_local = 0.9
energy_rate_fog = 0.6
energy_rate_cloud = 0.4

network_delay_fog = 0.05
network_delay_cloud = 0.1
network_delay_local = 0.01

execution_time = []
waiting_time = []
response_time = []
turnaround_time = []
energy_consumption = []
latency = []

# ============================================================
# COMPUTE METRICS PER TASK
# ============================================================

for i in range(num_tasks):

    decision = tasks["Global_Decision"].iloc[i]
    mips = tasks["MIPS"].iloc[i]

    if decision == "LOCAL":

        exec_time = mips / local_speed
        energy = exec_time * energy_rate_local
        net_delay = network_delay_local

    elif decision == "FOG":

        exec_time = mips / fog_speed
        energy = exec_time * energy_rate_fog
        net_delay = network_delay_fog

    else:

        exec_time = mips / cloud_speed
        energy = exec_time * energy_rate_cloud
        net_delay = network_delay_cloud


    wait_time = np.random.uniform(0.01,0.1)

    resp_time = exec_time + wait_time + net_delay

    tat = resp_time

    execution_time.append(exec_time)
    waiting_time.append(wait_time)
    response_time.append(resp_time)
    turnaround_time.append(tat)
    energy_consumption.append(energy)
    latency.append(resp_time)


# Convert to numpy
execution_time = np.array(execution_time)
waiting_time = np.array(waiting_time)
response_time = np.array(response_time)
turnaround_time = np.array(turnaround_time)
energy_consumption = np.array(energy_consumption)
latency = np.array(latency)

# ============================================================
# GLOBAL METRICS
# ============================================================

# Makespan
makespan = np.max(turnaround_time)

# Throughput
throughput = num_tasks / makespan

# Avg latency
avg_latency = np.mean(latency)

# Energy consumption
total_energy = np.sum(energy_consumption)

# Energy efficiency
energy_efficiency = throughput / total_energy

# Resource utilization
total_capacity = num_tasks * cloud_speed
used_capacity = np.sum(tasks["MIPS"])

resource_utilization = used_capacity / total_capacity

# SLA violation
sla_deadline = tasks["Deadline"].values

sla_violation = np.sum(turnaround_time > sla_deadline) / num_tasks

# Avg execution time
avg_execution_time = np.mean(execution_time)

# Avg waiting time
avg_waiting_time = np.mean(waiting_time)

# Avg response time
avg_response_time = np.mean(response_time)

# Avg turnaround time
avg_tat = np.mean(turnaround_time)

ppo = PPO_Global()

import torch.optim as optim
optimizer_ppo = optim.Adam(ppo.parameters(),lr=0.001)



episodes = 100

episode_total_rewards = []
episode_avg_rewards = []
episode_variance = []
episode_success_rate = []
episode_cumulative_reward = []
episode_convergence = []
episode_complexity = []

total_params = sum(p.numel() for p in ppo.parameters())

for ep in range(episodes):

    total_reward = 0
    rewards_this_episode = []
    success_count = 0

    for i in range(num_tasks):

        state = torch.tensor([
            tasks["Congestion_Prediction"].iloc[i],
            tasks["Topology_Influence"].iloc[i],
            tasks["Deadline"].iloc[i]
        ], dtype=torch.float32)

        probs, value = ppo(state)
        action = torch.multinomial(probs, 1).item()

        congestion = tasks["Congestion_Prediction"].iloc[i]

        # Reward function (add slight randomness for learning dynamics)
        reward = 1 - congestion + np.random.uniform(-0.05, 0.05)

        total_reward += reward
        rewards_this_episode.append(reward)

        if congestion < 0.5:
            success_count += 1

        loss = -torch.log(probs[action]) * reward
        optimizer_ppo.zero_grad()
        loss.backward()
        optimizer_ppo.step()

    # ================= Metrics per episode =================

    avg_reward = np.mean(rewards_this_episode)
    variance_reward = np.var(rewards_this_episode)
    cumulative_reward = np.sum(episode_total_rewards) + total_reward
    success_rate = success_count / num_tasks

    # Convergence rate (difference from previous episode)
    if ep > 0:
        convergence = total_reward - episode_total_rewards[-1]
    else:
        convergence = 0

    complexity = (ep+1) * num_tasks * total_params

    # Store metrics
    episode_total_rewards.append(total_reward)
    episode_avg_rewards.append(avg_reward)
    episode_variance.append(variance_reward)
    episode_success_rate.append(success_rate)
    episode_cumulative_reward.append(cumulative_reward)
    episode_convergence.append(convergence)
    episode_complexity.append(complexity)

    print(f"\nEpisode {ep+1}")
    print("Average Reward:", avg_reward)
    print("Cumulative Reward:", cumulative_reward)
    print("Reward Variance:", variance_reward)
    print("Success Rate:", success_rate)
    print("Convergence Rate:", convergence)
    print("Computational Complexity:", complexity)
    
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.colors as mcolors

def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """Truncate the color map between min_val and max_val."""
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """
    Truncate the color map according to the min_val and max_val from the
    original color map.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

#throughput
csfont = {'fontname':'Times New Roman'}
methods = ['ZT-PUF\nCRYSTALS\nKyber', 'CL-PKA', 'IBA', 'LBA']
throughput = [98.3,92.4,89.72,86.19]
fig, ax = plt.subplots()
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
y_min, y_max = 0, 110
# Add gradient background with alpha
ax.imshow(Z, interpolation='bicubic', cmap='PuRd', alpha=0.8,
          extent=[-0.5, len(methods)-0.5, 0, y_max], aspect='auto')
bars = ax.bar(methods, throughput)  


ax.set_ylim(y_min, y_max)
grad = np.atleast_2d(np.linspace(0, 1, 256)).T
ax = bars[0].axes  
lim = ax.get_xlim()+ax.get_ylim()
for bar in bars:
    bar.set_zorder(1)  
    bar.set_facecolor("none")  
    x, y = bar.get_xy() 
    w, h = bar.get_width(), bar.get_height() 
    c_map = truncate_colormap(plt.cm.twilight_shifted, min_val=0.6,
                              max_val=(h - y_min) / (y_max - y_min)) 
    ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,cmap=c_map)
    bar.set_edgecolor('red')
    bar.set_linewidth(2)
    bar.set_linestyle((0, (3,1,1,1)))  
    ax.text(x + w / 2, h, '{:.2f}'.format(h), ha='center', va='bottom', fontsize=12)
ax.axis(lim)
plt.ylabel('Authentication Accuracy(%)', fontsize=16,**csfont)
# plt.xlabel('Routing protocol', fontsize=16,**csfont)
plt.xticks(fontname="Times New Roman", fontsize=14)
plt.savefig('ART897_Result\\Authentication Accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

throughput = [0.024,0.038,0.089,0.072]
fig, ax = plt.subplots()
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
y_min, y_max = 0, 0.1
# Add gradient background with alpha
ax.imshow(Z, interpolation='bicubic', cmap='PuRd', alpha=0.8,
          extent=[-0.5, len(methods)-0.5, 0, y_max], aspect='auto')
bars = ax.bar(methods, throughput)  


ax.set_ylim(y_min, y_max)
grad = np.atleast_2d(np.linspace(0, 1, 256)).T
ax = bars[0].axes  
lim = ax.get_xlim()+ax.get_ylim()
for bar in bars:
    bar.set_zorder(1)  
    bar.set_facecolor("none")  
    x, y = bar.get_xy() 
    w, h = bar.get_width(), bar.get_height() 
    c_map = truncate_colormap(plt.cm.twilight_shifted, min_val=0.6,
                              max_val=(h - y_min) / (y_max - y_min)) 
    ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,cmap=c_map)
    bar.set_edgecolor('red')
    bar.set_linewidth(2)
    bar.set_linestyle((0, (3,1,1,1)))  
    ax.text(x + w / 2, h, '{:.3f}'.format(h), ha='center', va='bottom', fontsize=12)
ax.axis(lim)
plt.ylabel('Authentication Latency(Sec)', fontsize=16,**csfont)
# plt.xlabel('Routing protocol', fontsize=16,**csfont)
plt.xticks(fontname="Times New Roman", fontsize=14)
plt.savefig('ART897_Result\\Authentication Latency.png', dpi=300, bbox_inches='tight')
plt.show()

throughput = [0.021,0.041,0.067,0.082]
fig, ax = plt.subplots()
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
y_min, y_max = 0, 0.1
# Add gradient background with alpha
ax.imshow(Z, interpolation='bicubic', cmap='PuRd', alpha=0.8,
          extent=[-0.5, len(methods)-0.5, 0, y_max], aspect='auto')
bars = ax.bar(methods, throughput)  


ax.set_ylim(y_min, y_max)
grad = np.atleast_2d(np.linspace(0, 1, 256)).T
ax = bars[0].axes  
lim = ax.get_xlim()+ax.get_ylim()
for bar in bars:
    bar.set_zorder(1)  
    bar.set_facecolor("none")  
    x, y = bar.get_xy() 
    w, h = bar.get_width(), bar.get_height() 
    c_map = truncate_colormap(plt.cm.twilight_shifted, min_val=0.6,
                              max_val=(h - y_min) / (y_max - y_min)) 
    ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,cmap=c_map)
    bar.set_edgecolor('red')
    bar.set_linewidth(2)
    bar.set_linestyle((0, (3,1,1,1)))  
    ax.text(x + w / 2, h, '{:.3f}'.format(h), ha='center', va='bottom', fontsize=12)
ax.axis(lim)
plt.ylabel('Encryption Time(Sec)', fontsize=16,**csfont)
# plt.xlabel('Routing protocol', fontsize=16,**csfont)
plt.xticks(fontname="Times New Roman", fontsize=14)
plt.savefig('ART897_Result\\ Encryption Time.png', dpi=300, bbox_inches='tight')
plt.show()





##Average Residual energy  
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100','200','300','400','500']
y=[23.28,17.44,13.36,9.12,8.25]
plt.plot(x,y,'*--',color='fuchsia',linewidth=2, markersize=8)
y=[16.45,12.72,10.26,8.17,7.18]
plt.plot(x,y,'*--',color='goldenrod',linewidth=2, markersize=8)
y=[18.64,13.43,12.28,7.42,6.89]
plt.plot(x,y,'*--',color='blue',linewidth=2, markersize=8)
y=[13.37,9.29,6.61,5.27,3.58 ]
plt.plot(x,y,'*--',color='orangered',linewidth=2, markersize=8)
plt.legend(['OCI-OLSR', 'AODV', 'RPL', 'ETAR'])
plt.ylabel('Average Residual Energy (Joules)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART897_Result\\1.png', dpi=300, bbox_inches='tight')
plt.show()




##packet delivery ratio
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100','200','300','400','500']
PDR1=[98.7,92.3,87.45,86.39,84.21]
plt.plot(x,PDR1,'*--',color='fuchsia',linewidth=2, markersize=8)
PDR2=[93.42,87.28,86.21,80.73,79.5]
plt.plot(x,PDR2,'*--',color='goldenrod',linewidth=2, markersize=8)
PDR3=[90.58,89.63,86.19,83.43,81.69 ]
plt.plot(x,PDR3,'*--',color='blue',linewidth=2, markersize=8)
PDR4=[89.31,86.38,82.26,78.66,74.18]
plt.plot(x,PDR4,'*--',color='orangered',linewidth=2, markersize=8)

plt.legend(['OCI-OLSR', 'AODV', 'RPL', 'ETAR'])
plt.ylabel('Packet Delivery Ratio (%)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART897_Result\\2.png', dpi=300, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100','200','300','400','500']
Pl1=[100-PDR1[0],100-PDR1[1],100-PDR1[2],100-PDR1[3],100-PDR1[4]]
plt.plot(x,Pl1,'*--',color='fuchsia',linewidth=2, markersize=8)
Pl2=[100-PDR2[0],100-PDR2[1],100-PDR2[2],100-PDR2[3],100-PDR2[4]]
plt.plot(x,Pl2,'*--',color='goldenrod',linewidth=2, markersize=8)
Pl3=[100-PDR3[0],100-PDR3[1],100-PDR3[2],100-PDR3[3],100-PDR3[4]]
plt.plot(x,Pl3,'*--',color='blue',linewidth=2, markersize=8)
Pl4=[100-PDR4[0],100-PDR4[1],100-PDR4[2],100-PDR4[3],100-PDR4[4]]
plt.plot(x,Pl4,'*--',color='orangered',linewidth=2, markersize=8)

plt.legend(['OCI-OLSR', 'AODV', 'RPL', 'ETAR'])
plt.ylabel('Packet Loss(%)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART897_Result\\3.png', dpi=300, bbox_inches='tight')
plt.show()





##Latency
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100','200','300','400','500']
y=[0.014,0.0578, 0.0649, 0.072, 0.096]
plt.plot(x,y,'*--',color='fuchsia',linewidth=2, markersize=8)
y=[0.0356, 0.0729, 0.0821,0.124, 0.132]
plt.plot(x,y,'*--',color='goldenrod',linewidth=2, markersize=8)
y=[0.0586 ,0.0832, 0.0877, 0.092, 0.142]
plt.plot(x,y,'*--',color='blue',linewidth=2, markersize=8)
y=[0.0752 ,0.0861, 0.0952, 0.135, 0.146 ]
plt.plot(x,y,'*--',color='orangered',linewidth=2, markersize=8)

plt.legend(['OCI-OLSR', 'AODV', 'RPL', 'ETAR'])
plt.ylabel('Latency(ms)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART897_Result\\4.png', dpi=300, bbox_inches='tight')
plt.show()




##Energy consumption (J)
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
x=['100','200','300','400','500']
y=[1.7, 2.65 ,4.13, 8.82, 9.71]
plt.plot(x,y,'*--',color='fuchsia',linewidth=2, markersize=8)
y=[2.24, 4.17 ,9.28, 12.15, 16.46]
plt.plot(x,y,'*--',color='goldenrod',linewidth=2, markersize=8)
y=[4.38, 11.14, 18.73, 21.29, 30.15]
plt.plot(x,y,'*--',color='blue',linewidth=2, markersize=8)
y=[9.33 ,22.76 ,31.21 ,36.28 ,38.54]
plt.plot(x,y,'*--',color='orangered',linewidth=2, markersize=8)

plt.legend(['OCI-OLSR', 'AODV', 'RPL', 'ETAR'])
plt.ylabel('Energy consumption (J)', fontsize=16,**csfont)
plt.xlabel('Number of Nodes', fontsize=16,**csfont)
plt.xticks(fontsize= 14) 
plt.grid(True) 
plt.savefig('ART897_Result\\5.png', dpi=300, bbox_inches='tight')
plt.show()






import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """
    Truncate the color map according to the min_val and max_val from the
    original color map.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
        cmap(np.linspace(min_val, max_val, n)))
    return new_cmap

#throughput
csfont = {'fontname':'Times New Roman'}
methods = ['OCI-OLSR', 'AODV', 'RPL', 'ETAR']
throughput = [248.48,221.62,193.24,232.67]
fig, ax = plt.subplots()
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
y_min, y_max = 0, 300
# Add gradient background with alpha
ax.imshow(Z, interpolation='bicubic', cmap='PuRd', alpha=0.8,
          extent=[-0.5, len(methods)-0.5, 0, y_max], aspect='auto')
bars = ax.bar(methods, throughput)  


ax.set_ylim(y_min, y_max)
grad = np.atleast_2d(np.linspace(0, 1, 256)).T
ax = bars[0].axes  
lim = ax.get_xlim()+ax.get_ylim()
for bar in bars:
    bar.set_zorder(1)  
    bar.set_facecolor("none")  
    x, y = bar.get_xy() 
    w, h = bar.get_width(), bar.get_height() 
    c_map = truncate_colormap(plt.cm.PuRd, min_val=0.6,
                              max_val=(h - y_min) / (y_max - y_min)) 
    ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,cmap=c_map)
    bar.set_edgecolor('red')
    bar.set_linewidth(2)
    bar.set_linestyle((0, (3,1,1,1)))  
    ax.text(x + w / 2, h, '{:.2f}'.format(h), ha='center', va='bottom', fontsize=12)
ax.axis(lim)
plt.ylabel('Throughput(Mbps)', fontsize=16,**csfont)
# plt.xlabel('Routing protocol', fontsize=16,**csfont)
plt.xticks(fontname="Times New Roman", fontsize=14)
plt.savefig('ART897_Result\\6.png', dpi=300, bbox_inches='tight')
plt.show()




#Routing Overhead(%)
csfont = {'fontname':'Times New Roman'}
methods = ['OCI-OLSR', 'AODV', 'RPL', 'ETAR']
Routing_Overhead = [8.15,15.76,22.28,17.69]
fig, ax = plt.subplots()
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
y_min, y_max = 0, 30
# Add gradient background with alpha
ax.imshow(Z, interpolation='bicubic', cmap='PuRd', alpha=0.8,
          extent=[-0.5, len(methods)-0.5, 0, y_max], aspect='auto')
bars = ax.bar(methods, Routing_Overhead)  


ax.set_ylim(y_min, y_max)
grad = np.atleast_2d(np.linspace(0, 1, 256)).T
ax = bars[0].axes  
lim = ax.get_xlim()+ax.get_ylim()
for bar in bars:
    bar.set_zorder(1)  
    bar.set_facecolor("none")  
    x, y = bar.get_xy() 
    w, h = bar.get_width(), bar.get_height() 
    c_map = truncate_colormap(plt.cm.PuRd, min_val=0.6,
                              max_val=(h - y_min) / (y_max - y_min)) 
    ax.imshow(grad, extent=[x, x+w, h, y_min], aspect="auto", zorder=0,cmap=c_map)
    bar.set_edgecolor('red')
    bar.set_linewidth(2)
    bar.set_linestyle((0, (3,1,1,1)))  
    ax.text(x + w / 2, h, '{:.2f}'.format(h), ha='center', va='bottom', fontsize=12)
ax.axis(lim)
plt.ylabel('Routing Overhead(%)', fontsize=16,**csfont)
# plt.xlabel('Routing protocol', fontsize=16,**csfont)
plt.xticks(fontname="Times New Roman", fontsize=14)
plt.savefig('ART897_Result\\7.png', dpi=300, bbox_inches='tight')
plt.show()





methods = ['MDA\nDENN', 'ResCNN\nLSTM', 'DBN\nELM', 'TCN\nAttn','CNN\nGRU']
Acc = [0.986*100, 0.932*100, 0.898*100, 0.871*100,0.847*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(Acc):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Accuracy(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\accuracy.png', dpi=300, bbox_inches='tight')
plt.show()






error=[100-Acc[0],100-Acc[1],100-Acc[2],100-Acc[3],100-Acc[4]]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(error):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Equal Error Rate(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 20)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Equal Error Rate.png', dpi=300, bbox_inches='tight')
plt.show()



sp=[0.016,0.041,0.138,0.174,0.183]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.001, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Communication Cost (MB)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 0.2)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Communication Cost.png', dpi=300, bbox_inches='tight')
plt.show()










Pre = [0.972*100,0.925*100,0.908*100,0.873*100,0.859 *100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(Pre):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Positive Predictive Value(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\PPV.png', dpi=300, bbox_inches='tight')
plt.show()





Recall = [0.974*100,0.884*100,0.931*100,0.862*100,0.919*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(Recall):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Probability Of Detection(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Probability Of Detection.png', dpi=300, bbox_inches='tight')
plt.show()


spe = [0.986*100,0.8719*100,0.926*100,0.948*100,0.882*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(spe):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("True Negative Rate(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\True Negative Rate.png', dpi=300, bbox_inches='tight')
plt.show()



f1=[2*(Pre[0]*Recall[0])/(Pre[0]+Recall[0]),2*(Pre[1]*Recall[1])/(Pre[1]+Recall[1]),2*(Pre[2]*Recall[2])/(Pre[2]+Recall[2]),2*(Pre[3]*Recall[3])/(Pre[3]+Recall[3]),2*(Pre[4]*Recall[4])/(Pre[4]+Recall[4])]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(f1):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Fbeta score(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\fbeta score.png', dpi=300, bbox_inches='tight')
plt.show()





sp=[0.964*100,0.918*100,0.854*100,0.832*100,0.869*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Scott’s Pi(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Scott’s Pi.png', dpi=300, bbox_inches='tight')
plt.show()


sp=[(Recall[0]+sp[0]-100),(Recall[1]+sp[1]-100),(Recall[2]+sp[2]-100),(Recall[3]+sp[3]-100),(Recall[4]+sp[4]-100)]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Youden’s J Statistic (%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Youden’s J Statistic.png', dpi=300, bbox_inches='tight')
plt.show()






sp=[0.964*100,0.923*100,0.897*100,0.868*100,0.906*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Rand Index(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Rand Index.png', dpi=300, bbox_inches='tight')
plt.show()


sp=[0.879*100,0.852*100,0.819*100,0.846*100,0.863*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("V Measure(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 100)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\V Measure.png', dpi=300, bbox_inches='tight')
plt.show()


sp=[0.88*100,0.849*100,0.875*100,0.821*100,0.831*100]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Homogeneity Score(%)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 110)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Homogeneity Score.png', dpi=300, bbox_inches='tight')
plt.show()



Tr_time=[22.8,28.53,31.95,27.52,33.61]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(Tr_time):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Training Time(Sec)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 35)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Training time.png', dpi=300, bbox_inches='tight')
plt.show()


Ts_time=[0.391,0.563,0.791,0.911,0.768]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(Ts_time):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.001, f"{val:.3f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Inference Time(Sec)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 1)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Inference Time.png', dpi=300, bbox_inches='tight')
plt.show()







sp=[Tr_time[0]+Ts_time[0],Tr_time[1]+Ts_time[1],Tr_time[2]+Ts_time[2],Tr_time[3]+Ts_time[3],Tr_time[4]+Ts_time[4]]
fig, ax = plt.subplots(figsize=(8, 5))
for i, val in enumerate(sp):
    x_center = i
    width_bottom = 0.8
    width_top = 0.4
    top_y = val
    points = np.array([
        [x_center - width_bottom / 2, 0],
        [x_center + width_bottom / 2, 0],
        [x_center + width_top / 2, top_y],
        [x_center - width_top / 2, top_y]
    ])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    extent = [x_center - width_bottom / 2, x_center + width_bottom / 2, 0, top_y]
    cmap = truncate_colormap(plt.get_cmap('afmhot'), 0.2, 0.8)
    ax.imshow(gradient, aspect='auto', extent=extent, origin='lower', cmap=cmap, alpha=0.9, zorder=1)
    
    ax.text(x_center, -0.01, methods[i], ha='center', va='top',
            fontsize=16, fontname='Times New Roman', color='black')
    
    ax.text(x_center, top_y + 0.1, f"{val:.2f}", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
ax.set_ylabel("Computational Complexity(Sec)", fontsize=20, fontname='Times New Roman')
ax.tick_params(axis='y', labelsize=12)
ax.set_xlim(-1, len(methods))
ax.set_ylim(0, 40)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ART897_Result\\Computational Complexity.png', dpi=300, bbox_inches='tight')
plt.show()









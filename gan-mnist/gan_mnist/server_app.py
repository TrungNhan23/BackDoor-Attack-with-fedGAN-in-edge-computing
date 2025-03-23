# """gan-mnist: A Flower / PyTorch app."""

# from flwr.common import Context, ndarrays_to_parameters
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg

# from gan_mnist.task import Generator, Discriminator, GlobalModel, get_weights

# def fit_round(server_round: int):
#     """Truyền số round hiện tại xuống client."""
#     return {"current_round": server_round}

# def server_fn(context: Context):
#     # Read from config
#     num_rounds = context.run_config["num-server-rounds"]
#     fraction_fit = context.run_config["fraction-fit"]

#     # Initialize model parameters
#     # model =  GlobalModel()
#     D = Discriminator()
#     G = Generator()
    
#     ndarrays = get_weights(G) + get_weights(D)
#     parameters = ndarrays_to_parameters(ndarrays)

#     # Define strategy
#     strategy = FedAvg(
#         fraction_fit=fraction_fit,
#         fraction_evaluate=0.5,
#         initial_parameters=parameters, 
#         on_fit_config_fn=fit_round,
#         # fit_metrics_aggregation_fn=aggregate_fit_metrics
#     )
#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(strategy=strategy, config=config)

# # Create ServerApp  
# app = ServerApp(server_fn=server_fn)



"""gan-mnist: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple
from gan_mnist.task import Generator, Discriminator, GlobalModel, get_weights

def fit_round(server_round: int):
    """Truyền số round hiện tại xuống client."""
    return {"current_round": server_round}

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    D = Discriminator()
    # G = Generator()

    ndarrays = get_weights(D) 
    # + get_weights(G)
    parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.8, #evaluate all clients
        initial_parameters=parameters,
        on_fit_config_fn=fit_round,

    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
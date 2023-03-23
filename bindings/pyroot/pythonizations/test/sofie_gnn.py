import unittest
import ROOT

import numpy as np
from numpy.testing import assert_almost_equal
import graph_nets as gn
from graph_nets import utils_tf
import sonnet as snt



# for generating input data for graph nets,
# from https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb
def get_graph_data_dict(num_nodes, num_edges, GLOBAL_FEATURE_SIZE=2, NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2):
  return {
      "globals": np.random.rand(GLOBAL_FEATURE_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
  }

def CopyData(input_data) :
  output_data = ROOT.TMVA.Experimental.SOFIE.Copy(input_data)
  return output_data

class MLPGraphIndependent(snt.Module):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = gn.modules.GraphIndependent(
        edge_model_fn = lambda: snt.nets.MLP([2,2], activate_final=True),
        node_model_fn = lambda: snt.nets.MLP([2,2], activate_final=True),
        global_model_fn = lambda: snt.nets.MLP([2,2], activate_final=True))

  def __call__(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.Module):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))

  def __call__(self, inputs):
    return self._network(inputs)

class EncodeProcessDecode(snt.Module):

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    self._output_transform = MLPGraphIndependent()

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops

class SOFIE_GNN(unittest.TestCase):
    """
    Tests for the pythonizations of ParseFromMemory method of SOFIE GNN.
    """

    # def test_parse_gnn(self):
    #     '''
    #     Test that parsed GNN model from a graphnets model generates correct
    #     inference code
    #     '''
    #     GraphModule = gn.modules.GraphNetwork(
    #         edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
    #         node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
    #         global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))

    #     GraphData = get_graph_data_dict(2,1)
    #     input_graphs = utils_tf.data_dicts_to_graphs_tuple([GraphData])
    #     output = GraphModule(input_graphs)

    #     # Parsing model to RModel_GNN
    #     model = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(GraphModule, GraphData)
    #     model.Generate()
    #     model.OutputGenerated()

    #     ROOT.gInterpreter.Declare('#include "gnn_network.hxx"')
    #     input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()

    #     input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['nodes'])
    #     input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['edges'])
    #     input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['globals'])

    #     ROOT.TMVA_SOFIE_gnn_network.infer(input_data)

    #     output_node_data = output.nodes.numpy()
    #     output_edge_data = output.edges.numpy()
    #     output_global_data = output.globals.numpy().flatten()

    #     assert_almost_equal(output_node_data, np.asarray(input_data.node_data))
    #     assert_almost_equal(output_edge_data, np.asarray(input_data.edge_data))
    #     assert_almost_equal(output_global_data, np.asarray(input_data.global_data))


    # def test_parse_graph_independent(self):
    #     '''
    #     Test that parsed GraphIndependent model from a graphnets model generates correct
    #     inference code
    #     '''
    #     GraphModule = gn.modules.GraphIndependent(
    #         edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
    #         node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
    #         global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))

    #     GraphData = get_graph_data_dict(2,1)
    #     input_graphs = utils_tf.data_dicts_to_graphs_tuple([GraphData])
    #     output = GraphModule(input_graphs)

    #     # Parsing model to RModel_GraphIndependent
    #     model = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(GraphModule, GraphData)
    #     model.Generate()
    #     model.OutputGenerated()

    #     ROOT.gInterpreter.Declare('#include "graph_independent_network.hxx"')
    #     input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()

    #     input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['nodes'])
    #     input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['edges'])
    #     input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['globals'])

    #     ROOT.TMVA_SOFIE_graph_independent_network.infer(input_data)

    #     output_node_data = output.nodes.numpy()
    #     output_edge_data = output.edges.numpy()
    #     output_global_data = output.globals.numpy().flatten()

    #     assert_almost_equal(output_node_data, np.asarray(input_data.node_data))
    #     assert_almost_equal(output_edge_data, np.asarray(input_data.edge_data))
    #     assert_almost_equal(output_global_data, np.asarray(input_data.global_data))

    def test_lhcb_toy_inference(self):
        '''
        Test that parsed stack of SOFIE GNN and GraphIndependent modules generate the correct
        inference code
        '''
        # Instantiating EncodeProcessDecode Model
        ep_model = EncodeProcessDecode(2,2,2)

        # Initializing randomized input data
        GraphData = get_graph_data_dict(2,1)
        input_graphs = utils_tf.data_dicts_to_graphs_tuple([GraphData])

        # Initializing randomized input data for core
        CoreGraphData = get_graph_data_dict(2,1, 4, 4, 4)
        input_graphs_2 = utils_tf.data_dicts_to_graphs_tuple([CoreGraphData])

        # Collecting output from GraphNets model stack
        output_gn = ep_model(input_graphs, 2)

        # Declaring sofie models
        encoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._encoder._network, GraphData, filename = "encoder")
        encoder.Generate()
        encoder.OutputGenerated()

        core = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(ep_model._core._network, CoreGraphData, filename = "core")
        core.Generate()
        core.OutputGenerated()

        decoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._decoder._network, GraphData, filename = "decoder")
        decoder.Generate()
        decoder.OutputGenerated()

        output_transform = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._output_transform._network, GraphData, filename = "output_transform")
        output_transform.Generate()
        output_transform.OutputGenerated()

        # Including the sofie generated models
        ROOT.gInterpreter.Declare('#include "encoder.hxx"')
        ROOT.gInterpreter.Declare('#include "core.hxx"')
        ROOT.gInterpreter.Declare('#include "decoder.hxx"')
        ROOT.gInterpreter.Declare('#include "output_transform.hxx"')

        # Preparing the input data for running inference on sofie
        input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()
        input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['nodes'])
        input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['edges'])
        input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['globals'])

        # running inference on sofie
        ROOT.TMVA_SOFIE_encoder.infer(input_data)
        latent0 = CopyData(input_data)
        latent = input_data
        output_ops = []
        for _ in range(2):
            core_input = ROOT.TMVA.Experimental.SOFIE.Concatenate(latent0, latent, axis=1)
            ROOT.TMVA_SOFIE_core.infer(core_input)
            latent = CopyData(core_input)
            ROOT.TMVA_SOFIE_decoder.infer(core_input)
            ROOT.TMVA_SOFIE_output_transform.infer(core_input)
            output = CopyData(core_input)
            output_ops.append(output)

        for i in range(0, len(output_ops)):
          output_node_data = output_gn[i].nodes.numpy()
          output_edge_data = output_gn[i].edges.numpy()
          output_global_data = output_gn[i].globals.numpy().flatten()

          assert_almost_equal(output_node_data, np.asarray(output_ops[i].node_data))
          assert_almost_equal(output_edge_data, np.asarray(output_ops[i].edge_data))
          assert_almost_equal(output_global_data, np.asarray(output_ops[i].global_data))




if __name__ == '__main__':
    unittest.main()

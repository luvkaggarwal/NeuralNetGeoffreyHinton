function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    visible_data = sample_bernoulli(visible_data);
    hidden_data = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w,visible_data));
    visible_reconstruction = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w,hidden_data));
    hidden_reconstruction = visible_state_to_hidden_probabilities(rbm_w,visible_reconstruction);
    data_goodness = configuration_goodness_gradient(visible_data,hidden_data);
    reconstruction_goodness = configuration_goodness_gradient(visible_reconstruction,hidden_reconstruction);
    ret = data_goodness .- reconstruction_goodness;
end

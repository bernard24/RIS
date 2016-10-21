
NAME=baladre
MODELS_DIR=../../../Torch_models/
MODELS_LSTM_DIR=../../../Torch_models/
CONVLSTM=$MODELS_LSTM_DIR/coco_convlstm.model
FCN8_1=$MODELS_DIR/coco_fcn8_1.model
FCN8_2=$MODELS_DIR/coco_fcn8_2.model
th experiment.lua -learning_rate 0.00001 -train_or_val all -rnn_channels 100 -rnn_layers 2 -rnn_filter_size 1 -fcn8_1_model $FCN8_1 -fcn8_2_model $FCN8_2 -lstm_model $CONVLSTM -it 1000000 -summary_after 1000 -learn_fcn8_1 0 -learn_fcn8_2 1 -learn_lstm 1 -learn_shall_we_stop 0 -gpu_setDevice 0 -class 15 -lambda 1 -non_object_iterations 1 -seq_length 10 -name $NAME

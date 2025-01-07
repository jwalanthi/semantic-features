#! /bin/sh
cmd="python model.py" 
echo "Hi there, I'm here to help you train some models! I will train a classifier for each layer of the lm, keeping all other factors the same. Which norm would you like to use?"
read norm
cmd+=" --norm=$norm"
echo "File which has the norms?"
read norm_file
cmd+=" --norm_file=$norm_file"
echo "Embedding directory?"
read embedding_dir
cmd+=" --embedding_dir=$embedding_dir"
echo "How many layers are in this LM?"
read lm_layers

echo "Save directory?"
read save_dir
cmd+=" --save_dir=$save_dir"
echo "Save model name?"
read save_model_name

echo "OK! Now for some optional parameters"

echo "Optimize? Y/n"
read response
if [ $response == "Y" ]
then 
    cmd+=" --optimize"
fi
echo "Prune? Y/n"
read response
if [ $response == "Y" ]
then 
    cmd+=" --prune"
fi
echo "If using GPU for optimization, which device?"
read gpu
if [[ $gpu != "" ]]
then 
    cmd+=" --gpu=$gpu"
fi

echo "Number of Layers in Classifier?"
read num_layers
if [[ $num_layers != "" ]]
then 
    cmd+=" --num_layers=$num_layers"
fi
echo "Hidden size?"
read hidden_size
if [[ $hidden_size != "" ]]
then 
    cmd+=" --hidden_size=$hidden_size"
fi
echo "Dropout rate?"
read dropout
if [[ $dropout != "" ]]
then 
    cmd+=" --dropout=$dropout"
fi
echo "Max Epochs?"
read num_epochs
if [[ $num_epochs != "" ]]
then 
    cmd+=" --num_epochs=$num_epochs"
fi
echo "Batch size?"
read 
if [[ $batch_size != "" ]]
then 
    cmd+=" --batch_size=$batch_size"
fi
echo "Learning Rate?"
read learning_rate
if [[ $learning_rate != "" ]]
then 
    cmd+=" --learning_rate=$learning_rate"
fi
echo "Weight decay?"
read weight_decay
if [[ $weight_decay != "" ]]
then 
    cmd+=" --weight_decay=$weight_decay"
fi
echo "Early stopping, how many epochs to wait?"
read early_stopping
if [[ $early_stopping != "" ]]
then 
    cmd+=" --early_stopping=$early_stopping"
fi

echo "use raw buchanan values? Y/n"
read response
if [ $response == "Y" ]
then 
    cmd+=" --raw_buchanan"
fi
echo "Use normalized buchanan values? Y/n"
read response
if [ $response == "Y" ]
then 
    cmd+=" --normal_buchanan"
fi

i=0
lm_layers=`expr $lm_layers + 1`
while [ $i -lt $lm_layers ]
do 
    ncmd=$cmd" --lm_layer=$i --save_model_name=$save_model_name""_layer$i"
    echo "your command is $ncmd"
    eval "$ncmd"
    i=`expr $i + 1`
done
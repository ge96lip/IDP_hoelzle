
#!/bin/bash
# Activate the virtual environment
source /Users/carlottaholzle/Desktop/SS2024/IDP/.venv/bin/activate

export PYTHONPATH=./:$PYTHONPATH
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"

# Extracting the positional arguments
task=$1
output_dir=$2
data_dir=$3

# Shift the first three arguments so that we can process the optional ones
shift 3
# Append each optional argument to the command
echo $task
if [ "$task" == "eval" ]; then
    site_names_file=$1
    shift 1
fi


# Initialize an array to hold the optional arguments
optional_args=()

# Iterate over the remaining arguments and add them to the optional_args array
while [[ "$#" -gt 0 ]]; do
    optional_args+=("$1")
    shift
done

# Set the file to be used based on the task
if [ "$task" == "train" ]; then
    file="train.py"
    cmd="python $file --out_dir=$output_dir --data_dir=$data_dir"
elif [ "$task" == "eval" ]; then
    file="eval.py"
    cmd="python $file --out_dir=$output_dir --data_dir=$data_dir --site_names_file=$site_names_file"
else
    echo "Unknown task: $task. Task must be 'train' or 'eval'."
    exit 1
fi

# Construct the command

# Construct the command
for arg in "${optional_args[@]}"; do
    # Quote the argument if it contains special characters
    if [[ "$arg" == *[\{\}]* ]]; then
        echo $arg
        cmd="$cmd \"$arg\""
    else
        cmd="$cmd $arg"
    fi
done

# Execute the command
eval $cmd

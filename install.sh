# Confirm user has Python 3 installed
p=''
if [ -z $(command -v python3) ]; then
    # python3 command does not exist
    if [ -z $(command -v python) ]; then
        # python command doesn't exist either, ask them to install it
        echo 'Please install Python version 3.5 or greater'
        exit
    else
        # python command exists, need to check version
        pythonversion=$(python -c 'import sys; print(int(float("{}.{}".format(sys.version_info.major, sys.version_info.minor))*10))')
        if [ $pythonversion -lt 35 ]; then
            # Python version not high enough
            echo "Current Python version: $pythonversion"
            echo 'Please install Python version 3.5 or greater'
            exit
        fi
    fi
else
    # python3 command exists, we're good to go
    p='3'
fi
echo 'Python version check successful'

python$p -m venv venv # Create virtual environment
source venv/bin/activate # Activate virtual environment
pip$p install -r requirements.txt # Install dependencies
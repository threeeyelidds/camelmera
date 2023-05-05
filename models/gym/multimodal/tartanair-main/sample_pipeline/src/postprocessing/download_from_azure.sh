ENVNAME="OldScandinaviaExposure"

ENV="/ocean/projects/cis220039p/shared/tartanair_v2/"${ENVNAME}
SOURCE="https://tartanairv2.blob.core.windows.net/data-raw/"${ENVNAME}
TOCKEN="sv=2021-04-10&st=2023-03-24T13%3A54%3A41Z&se=2023-04-30T13%3A54%3A00Z&sr=c&sp=racwdxltf&sig=ysAbUStRig2DjKKA11cnivjQhGp2INo7O17LJkfYpuw%3D"

TRJS="P000 P001 P002 P003 P004 P005 P006 P007 P008 P009 P010 P011 P012 P013 P014 P015 P016 P017 P018 P019 P020 P021"
DATAS="Data_easy Data_hard"

if [ -d ${ENV} ]
then
    echo "Directory ${ENV} exists." 
else
    mkdir ${ENV}
fi

for DD in ${DATAS}
do
    echo ${DD}
    for TRJNO in $TRJS  
    do
        echo ${TRJNO}
        
        if [ -d ${ENV}/${DD} ]
        then
            echo "Directory ${ENV}/${DD} exists." 
        else
            mkdir ${ENV}/${DD}
        fi

        echo
        echo ======
        echo "${SOURCE}/${DD}/${TRJNO}"

        ./azcopy copy "${SOURCE}/${DD}/${TRJNO}?${TOCKEN}" ${ENV}/${DD}/ --recursive --as-subdir=true

    done
done
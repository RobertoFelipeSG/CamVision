rename 's/.+/our $i; sprintf("%03d.jpg", 1+$i++)/e' * -vn

find ./negative_images -iname "*.jpg" > negatives.txt

convert -resize 50% myfigure.png myfigure.jpg
find . -name "*.jpg" | xargs mogrify -resize 200%

opencv_createsamples -img 13.jpg -bg negatives.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950
opencv_createsamples -img 13.jpg -bg negatives.txt -info info/info.lst -pngoutput info -bgcolor 0 -bgthresh 0 -maxxangle 1.1  -maxyangle 1.1 maxzangle 0.5 -num 1950 -maxidev 40 -w 200 -h 100

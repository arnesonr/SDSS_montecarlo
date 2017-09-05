set terminal pdf
set output "uniform.pdf"
set label 1 "\"I flew here... by plane.  Why?  For the halibut.\"" at 64, 135, 0 centre norotate back nopoint offset character 0, 0, 0
set title "Larry Ewing's GIMP penguin on vacation basking in\nthe balmy waters off the coast of Murmansk" 
set xrange [ -10.0000 : 137.000 ] noreverse nowriteback
set yrange [ -10.0000 : 157.000 ] noreverse nowriteback
set palette gray
set colorbox
plot 'blutux.rgb' binary array=10x25 flipy format=%uchar%uchar%uchar' using ($1+$2+$3)/3 with image

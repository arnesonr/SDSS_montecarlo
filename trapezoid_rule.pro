function trapezoid_rule, y,a,b,n
;n is the subdivision per unit (i.e. n=5 has divisions of 0.2 etc...)
;x=dindgen(n*(b-a))/n+a

return, ((b-a)/((b-a)*n))*((1d/2d)*y[0] + (1d/2d)*y[(b-a)*n-1] + total(y[1:(b-a)*n-2]))

end

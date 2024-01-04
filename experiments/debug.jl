using Enzyme
@info "starting"
f1(x) = x*x
autodiff(Reverse, f1, Active(1.0))
@info "done"

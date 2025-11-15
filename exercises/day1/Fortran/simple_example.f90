
subroutine f(u, x)
  real, intent(out) :: u
  real, intent(in)  :: x(3)
  u = 3*x(1) + 2*x(2) + x(3)
end subroutine f


subroutine f1(y, x)
  real, intent(out) :: y(2)
  real, intent(in)  :: x(3)
  real :: w, v, u
  real, parameter :: pi = 3.14159265358979320
  u = 3*x(1) + 2*x(2) + x(3)
  w = pi * sin(u)
  v = pi * cos(u)
  y(1) = v * w
  y(2) = w * x(1)
end subroutine f1



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
  v = pi * sin(u)
  w = pi * cos(u) * v
  y(1) = w
  y(2) = v
end subroutine f1


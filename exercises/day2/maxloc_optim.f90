subroutine foo(n, x, y)
  implicit none
  integer :: n
  real :: x(n)
  real :: y(n)
  integer :: i, iloc_array(1), iloc

  iloc_array = maxloc(x)
  iloc = iloc_array(1)

  !$AD II-LOOP
  do i = 1, n
    y(i) = x(i) / x(iloc)
  end do
end subroutine foo


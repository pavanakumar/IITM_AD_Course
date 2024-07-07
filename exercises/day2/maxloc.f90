subroutine foo(n, x, y)
  implicit none
  integer :: n
  real :: x(n)
  real :: y(n)
  integer :: i
  real :: max_value

  max_value = x(1)

  do i = 1, n
    if(max_value .lt. x(i)) &
      max_value = x(i)
  end do

  do i = 1, n
    y(i) = x(i) / max_value
  end do

end subroutine foo


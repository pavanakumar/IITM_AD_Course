program run_stencil
  implicit none
  integer, parameter :: n = 5
  real :: u(n), ud(n)
  real :: res(n), resd(n)
  integer :: i

  u = 1.0
  res = 0.0
  do i = 1, n
    ud = 0.0
    ud(i) = 1.0
    call stencil_d(n, (/ 1.0, -2.0, 3.0 /), u, ud, res, resd)
    write(*,*) resd
  end do
 
end program run_stencil


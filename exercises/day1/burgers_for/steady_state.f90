! implicit root finiding viscous Burgers' solution
program steady_state
  implicit none
  real(8) :: nu, A, u, x, x0
  real(8) :: delta
  character(len=1024) :: arg
  integer :: i, n
  real(8), allocatable :: u_sol(:)
 
  ! get nu value
  call get_command_argument(1, arg)
  read(arg, *, END=111, ERR=111) nu

  ! get n
  call get_command_argument(2, arg)
  read(arg, *, END=111, ERR=111) n

  ! get delta
  call get_command_argument(3, arg)
  read(arg, *, END=111, ERR=111) delta

  allocate( u_sol(n) )
  A = 2.00d0
  call find_root( nu, delta, A, 1.0e-12, 10000 )
  call objective (nu, A, x0)
  if ( delta .eq. 0.00d0 ) x0 = 0.00d0
 
  do i = 1, n
    x = -1.00d0 + (i - 1) * 2.00d0 / n
    call solution( nu, A, x0, x, u )
    u_sol(i) = u
  end do

  do i = 1, n
    write(*, *) -1.00d0 + (i - 1) * 2.00d0 / n, u_sol(i)
  end do
  write(*,*) 1.00d0, -1.00d0
  return

111 print *, 'Error: In command line args'  
    print *, 'Specify steady_state_xx <nu> <n> <delta>'
  stop
end program steady_state

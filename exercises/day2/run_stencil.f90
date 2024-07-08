program run_stencil
  implicit none
  integer, parameter :: n = 5
  real :: u(n), ud(n)
  real :: res(n), resd(n)
  real :: jacob(n, n)
  real :: jacob_colour(n, 3)
  integer :: i, j
  integer :: icolour

  do i = 1, n
    u = 1.0
    res = 0.0
    ud = 0.0
    resd = 0.0
    ud(i) = 1.0
    call stencil_d(n, (/ 1.0, -2.0, 3.0 /), u, ud, res, resd)
    ! Construct one column at a time
    jacob(i, 1:n) = resd
  end do

  write(*,*)
  write(*, *) "Forward mode Jacobian (A x)"
  write(*, *) "--------------------------"
  do i = 1, n
   do j = 1, n
     write(*, "(SP, F8.2,A)", advance='no') jacob(j, i) 
   end do
   write(*,*)
  end do 

  write(*,*) 
  write(*, *) "Reverse mode Jacobian (A^T x)"
  write(*, *) "-----------------------------"
  do i = 1, n
   do j = 1, n
     write(*, "(SP, F8.2,A)", advance='no') jacob(i, j) 
   end do
   write(*,*)
  end do 

  ! Seed based on 3 colouring
  write(*,*)
  write(*,*) "Seed vectors"
  write(*,*) "------------"
  do icolour = 1, 3
    ud = 0.0
    do i = icolour, n, 3
      ud(i) = 1.0
    end do
    write(*,"(A,I1,A)", advance='no') "   Seed (", icolour, ")"
    do i = 1, n
      write(*, "(SP, F8.2,A)", advance='no') ud(i)
    end do
    write(*,*)
  end do 

  ! Jacobian colouring (3 colour)
  do icolour = 1, 3
    u = 1.0
    res = 0.0
    ud = 0.0
    resd = 0.0
    ! Seed based on 3 colouring
    do i = icolour, n, 3
      ud(i) = 1.0
    end do
    call stencil_d(n, (/ 1.0, -2.0, 3.0 /), u, ud, res, resd)
    ! Construct one colour at a time
    jacob_colour(1:n, icolour) = resd(1:n)
  end do

  write(*,*) 
  write(*,*) "Jacobian Coloured"
  write(*,*) "-----------------"
  do i = 1, n
    do icolour = 1, 3
      write(*, "(SP, F8.2,A)", advance='no') jacob_colour(i, icolour)
    end do
    write(*,*)
  end do

end program run_stencil


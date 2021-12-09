module functions
  implicit none

contains
  pure function generate_fourier_term_matrix(x,R) result(out)
    real, dimension(:), intent(in) :: x
    integer, intent(in) :: R
    real, dimension(2*R+1,2*R+1) :: out

    real, dimension(2*R+1,2*R+1) :: series
    integer :: i,j

    do j=1,2*R+1
       do i=1,2*R+1
          if(j==1)then
             series(i,j)=1
          elseif(mod(i,2)==1)then
             series(i,j) = sin((j-1)/2 * x(i))
          else
             series(i,j) = cos(j/2 * x(i))
          end if
       end do
    end do
    out = series
  end function generate_fourier_term_matrix

end module functions


program main
  use functions
  implicit none

  integer :: i

  write (*,*) generate_fourier_term_matrix(real([(i, i=-2,2)]), 2)
end program main

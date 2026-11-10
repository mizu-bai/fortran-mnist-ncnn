module ncnn_mnist
    !> Thin wrapper over the C++ shim in predict.cpp.
    !>
    !> predict() takes a pointer to 28*28 floats laid out in ROW-MAJOR order,
    !> which is what ncnn expects -- not Fortran's column-major. main.f90 is
    !> responsible for producing that layout; see the comment there.
    use, intrinsic :: iso_c_binding, only: c_float
    implicit none

    interface
        subroutine predict(image, label_pred) bind(c, name="predict")
            import :: c_float
            implicit none

            ! explicit-shape dummy in a bind(c) interface is passed by
            ! reference, so C receives a const float* -- matching predict.h
            real(c_float), intent(in) :: image(28, 28)
            ! no value attribute: passed by reference, so C receives an int*
            integer, intent(out) :: label_pred
        end subroutine predict
    end interface

end module ncnn_mnist

program main
    !> Runs mnist-12 on a 28x28 image stored as 28 rows of 28 ASCII numbers.
    !>
    !>   ./mnist image/4.dat
    use ncnn_mnist, only: predict
    use, intrinsic :: iso_c_binding, only: c_float
    implicit none

    real(c_float) :: image(28, 28)
    character(len=4096) :: image_name
    character(len=1024) :: line
    character(len=256)  :: errmsg
    integer :: i, label_pred, argc, ios, unit

    !> parse input
    argc = command_argument_count()

    if (argc /= 1) then
        print *, "Usage: ./mnist image.dat"
        stop 1
    end if

    call get_command_argument(1, image_name)

    !> open file -- check it actually opened, otherwise image would stay
    !> uninitialised and the prediction would be meaningless
    open (newunit=unit, file=trim(image_name), status="old", action="read", &
          iostat=ios, iomsg=errmsg)
    if (ios /= 0) then
        print *, "Cannot open ", trim(image_name), ": ", trim(errmsg)
        stop 1
    end if

    !> read image, one scanline per row
    !>
    !> Each line is read into a buffer and then parsed from there, so a line
    !> with fewer than 28 values fails instead of silently swallowing the next
    !> line -- list-directed read straight from the unit would do that.
    !>
    !> Row i of the file is stored as column i of the Fortran array, which
    !> makes the underlying memory row-major: the layout ncnn::Mat expects,
    !> not Fortran's native column-major.
    do i = 1, 28
        read (unit, '(A)', iostat=ios, iomsg=errmsg) line
        if (ios /= 0) then
            print *, "Read failed on row ", i, " of ", trim(image_name), &
                     ": ", trim(errmsg)
            close (unit)
            stop 1
        end if

        read (line, *, iostat=ios, iomsg=errmsg) image(:, i)
        if (ios /= 0) then
            print *, "Row ", i, " of ", trim(image_name), " does not hold 28 values: ", &
                     trim(errmsg)
            close (unit)
            stop 1
        end if
    end do
    close (unit)

    !> call ncnn
    call predict(image, label_pred)

    !> result
    if (label_pred < 0) then
        print *, "Predict: failed (see stderr)"
        stop 1
    end if

    print *, "Predict: ", label_pred

end program main

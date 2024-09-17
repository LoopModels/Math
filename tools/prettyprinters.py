from __future__ import print_function

import gdb.printing


class BaseVectorPrinter(Iterator):
    """Print a Vector<>"""

    def __init__(self, ptr, size):
        self.begin = ptr
        self.size = size
        self.i = 0

    def __next__(self):
        if self.i == self.size:
            raise StopIteration
        ret = "[{}]".format(self.i), (self.begin + self.i).dereference()
        self.i += 1
        return ret

    def to_string(self):
        return "Vector of size: {}".format(self.size)

    def display_hint(self):
        return "array"

class VectorPrinter(BaseVectorPrinter):
    def __init__(self, val):
        t = val.type.template_argument(0).pointer()
        BaseVectorPrinter.__init__(self, val["ptr"].cast(t), val["sz"]["value_"].cast(gdb.lookup_type("long")))

class TinyVectorPrinter(BaseVectorPrinter):
    def __init__(self, val):
        t = val.type.template_argument(0).pointer()
        BaseVectorPrinter.__init__(self, val["data_"]["mem"][0].address.cast(t), val["len_"]["value_"].cast(gdb.lookup_type("long")))

class BaseMatrixPrinter:
    """Print a StridedMatrix<>"""

    def __init__(self, begin, rows, cols, stride):
        self.begin = begin
        self.rows = rows
        self.cols = cols
        self.stride = stride

    def to_string(self):
        header = "Matrix, {} x {}, stride {}:\n".format(
            self.rows, self.cols, self.stride
        )
        s = [
            [
                str((self.begin + (c + r * self.stride)).dereference())
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]
        # header += len(s[0][0])
        lens = [max(len(val) for val in col) for col in zip(*s)]
        # gdb.write("Lens: {}, {}, {}".format(lens[0], lens[1], lens[2]))
        try:
            fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        except Exception as exc:
            gdb.write("Exception: {}\n".format(exc))
        table = [fmt.format(*row) for row in s]
        return header + '\n'.join(table)

class SquareMatrixPrinter(BaseMatrixPrinter):
    """Print a Matrix<>"""

    def __init__(self, val):
        t = val.type.template_argument(0).pointer()
        M = val["sz"]["m_"]["value_"].cast(gdb.lookup_type("long"))
        BaseMatrixPrinter.__init__(self, val["ptr"].cast(t), M, M, M)


class DenseMatrixPrinter(BaseMatrixPrinter):
    """Print a Matrix<>"""

    def __init__(self, val):
        t = val.type.template_argument(0).pointer()
        M = val["sz"]["m_"]["value_"].cast(gdb.lookup_type("long"))
        N = val["sz"]["n_"]["value_"].cast(gdb.lookup_type("long"))
        BaseMatrixPrinter.__init__(self, val["ptr"].cast(t), M, N, N)


class StridedMatrixPrinter(BaseMatrixPrinter):
    """Print a Matrix<>"""

    def __init__(self, val):
        t = val.type.template_argument(0).pointer()
        BaseMatrixPrinter.__init__(
            self,
            val["ptr"].cast(t),
            val["sz"]["m_"]["value_"].cast(gdb.lookup_type("long")),
            val["sz"]["n_"]["value_"].cast(gdb.lookup_type("long")),
            val["sz"]["stride_m_"]["value_"].cast(gdb.lookup_type("long")),
        )


class WrappedIntegerPrinter:
    """Print an integer wrapped in a struct."""

    def __init__(self, val):
        self.val = val["value_"].cast(gdb.lookup_type("long"))

    def to_string(self):
        return str(self.val)

pp = gdb.printing.RegexpCollectionPrettyPrinter("PolyMath")
pp.add_printer(
    "math::PtrVector",
    "^math::(Mut)?Array<.*, math::Length<-1[l]?, long>,[^,]*?>$",
    VectorPrinter,
)
pp.add_printer(
    "math::Vector",
    "^math::ManagedArray<.*, math::Length<-1[l]?, long>,[^,]*?,[^,]*?>$",
    VectorPrinter,
)
pp.add_printer(
    "containers::TinyVector",
    "^containers::TinyVector<.*, .*,[^,]*?>$",
    TinyVectorPrinter,
)
pp.add_printer(
    "math::SquarePtrMatrix",
    "^math::(Mut)?Array<.*, math::SquareDims<-1[l]?>,[^,]*?>$",
    SquareMatrixPrinter,
)
pp.add_printer(
    "math::DensePtrMatrix",
    "^math::(Mut)?Array<.*, math::DenseDims<-1[l]?, -1[l]?>,[^,]*?>$",
    DenseMatrixPrinter,
)
pp.add_printer(
    "math::StridedPtrMatrix",
    "^math::(Mut)?Array<.*, math::StridedDims<-1[l]?, -1[l]?, -1[l]?>,[^,]*?>$",
    StridedMatrixPrinter,
)
pp.add_printer(
    "math::SquareMatrix",
    "^math::ManagedArray<.*, math::SquareDims<-1[l]?>,[^,]*?,[^,]*?>$",
    SquareMatrixPrinter,
)
pp.add_printer(
    "math::DenseMatrix",
    "^math::ManagedArray<.*, math::DenseDims<-1[l]?, -1[l]?>,[^,]*?,[^,]*?>$",
    DenseMatrixPrinter,
)
pp.add_printer(
    "math::StridedMatrix",
    "^math::ManagedArray<.*, math::StridedDims<-1[l]?, -1[l]?, -1[l]?>,[^,]*?,[^,]*?>$",
    StridedMatrixPrinter,
)
pp.add_printer(
    "math::Length",
    "^math::Length<-1[l]?, long>$",
    WrappedIntegerPrinter,
)
pp.add_printer(
    "math::Capacity",
    "^math::Capacity<-1[l]?, long>$",
    WrappedIntegerPrinter,
)
pp.add_printer(
    "math::Row", "^math::Row<-1[l]?>$", WrappedIntegerPrinter
)
pp.add_printer(
    "math::Col", "^math::Col<-1[l]?>$", WrappedIntegerPrinter
)
pp.add_printer(
    "math::RowStride",
    "^math::RowStride<-1[l]?>$",
    WrappedIntegerPrinter,
)

gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)

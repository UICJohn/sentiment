from orator.migrations import Migration


class CreateEmMatrixTable(Migration):

    def up(self):
        with self.schema.create('em_matrices') as table:
            table.increments('id')
            table.string('word')
            table.text('vector')
            table.index('word')
            table.index('vector')
            table.timestamps()

    def down(self):
        self.schema.drop('em_matrix')

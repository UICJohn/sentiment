from orator.migrations import Migration


class ChangeColumnPositiveToLabel(Migration):

    def up(self):
        with self.schema.table('training_sets') as table:
            table.rename_column('positive','label')

    def down(self):
        """
        Revert the migrations.
        """
        pass


from orator.migrations import Migration


class CreateIndexForTrainingSetTables(Migration):

  def up(self):
    with self.schema.table('training_sets') as table:
      table.index('iterations')      
      
  def down(self):
    """
    Revert the migrations.
    """
    pass

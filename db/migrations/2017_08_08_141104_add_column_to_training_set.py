from orator.migrations import Migration


class AddColumnToTrainingSet(Migration):

  def up(self):
    with self.schema.table('training_sets') as table:
      table.boolean('trained').default(False)

  def down(self):
      """
      Revert the migrations.
      """
      pass

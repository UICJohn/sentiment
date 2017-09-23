from orator.migrations import Migration


class TweakTrainingSetTable(Migration):

  def up(self):
    with self.schema.table('training_sets') as table:
      table.integer('iterations').default(1)
      table.drop_column('trained')
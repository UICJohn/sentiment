from orator.migrations import Migration


class CreateTrainingSetsTable(Migration):

  def up(self):
    with self.schema.create('training_sets') as table:
      table.increments('id')
      table.text('words')
      table.integer("positive")
      table.text('word_ids').nullable()
      table.index('positive')
      table.timestamps()

  def down(self):
    """
    Revert the migrations.
    """
    pass

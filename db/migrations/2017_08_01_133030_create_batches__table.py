from orator.migrations import Migration


class CreateBatches_table(Migration):

  def up(self):
    with self.schema.create('batches') as table:
      table.increments('id')
      table.integer('set_id')
      table.timestamps()

  def down(self):
      """
      Revert the migrations.
      """
      pass

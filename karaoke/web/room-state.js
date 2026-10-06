// Room revisions order commands; progress and cache updates can share a revision.
export function isStaleRoomSnapshot(incoming, current) {
  return incoming.revision < current.revision ||
    (incoming.revision === current.revision && incoming.server_time < current.server_time);
}

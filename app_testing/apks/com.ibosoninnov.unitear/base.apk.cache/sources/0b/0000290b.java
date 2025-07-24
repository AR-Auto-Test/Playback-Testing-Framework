package com.google.mediapipe.framework;

import com.google.common.base.Preconditions;
import com.google.common.flogger.FluentLogger;
import com.google.mediapipe.proto.CalculatorProto;
import com.google.mediapipe.proto.GraphTemplateProto;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/Graph.class */
public class Graph {
    private static final FluentLogger logger = FluentLogger.forEnclosingClass();
    private static final int MAX_BUFFER_SIZE = 20;
    private final List<PacketCallback> packetCallbacks = new ArrayList();
    private Map<String, Packet> sidePackets = new HashMap();
    private Map<String, Packet> streamHeaders = new HashMap();
    private boolean stepMode = false;
    private boolean startRunningGraphCalled = false;
    private boolean graphRunning = false;
    private Map<String, ArrayList<PacketBufferItem>> packetBuffers = new HashMap();
    private final Object terminationLock = new Object();
    private long nativeGraphHandle = nativeCreateGraph();

    private native long nativeCreateGraph();

    private native void nativeReleaseGraph(long context);

    private native void nativeAddPacketCallback(long context, String streamName, PacketCallback callback);

    private native long nativeAddSurfaceOutput(long context, String streamName);

    private native void nativeLoadBinaryGraph(long context, String path);

    private native void nativeLoadBinaryGraphBytes(long context, byte[] data);

    private native void nativeLoadBinaryGraphTemplate(long context, byte[] data);

    private native void nativeSetGraphType(long context, String graphType);

    private native void nativeSetGraphOptions(long context, byte[] data);

    private native byte[] nativeGetCalculatorGraphConfig(long context);

    private native void nativeRunGraphUntilClose(long context, String[] streamNames, long[] packets);

    private native void nativeStartRunningGraph(long context, String[] sidePacketNames, long[] sidePacketHandles, String[] streamNamesWithHeader, long[] streamHeaderHandles);

    private native void nativeAddPacketToInputStream(long context, String streamName, long packet, long timestamp);

    private native void nativeMovePacketToInputStream(long context, String streamName, long packet, long timestamp);

    private native void nativeSetGraphInputStreamBlockingMode(long context, boolean mode);

    private native void nativeCloseInputStream(long context, String streamName);

    private native void nativeCloseAllInputStreams(long context);

    private native void nativeCloseAllPacketSources(long context);

    private native void nativeWaitUntilGraphDone(long context);

    private native void nativeWaitUntilGraphIdle(long context);

    private native void nativeUpdatePacketReference(long referencePacket, long newPacket);

    private native void nativeSetParentGlContext(long context, long javaGlContext);

    private native void nativeCancelGraph(long context);

    private native long nativeGetProfiler(long context);

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/Graph$PacketBufferItem.class */
    public static class PacketBufferItem {
        final Packet packet;
        final Long timestamp;

        private PacketBufferItem(Packet packet, Long timestamp) {
            this.packet = packet;
            this.timestamp = timestamp;
        }
    }

    public synchronized long getNativeHandle() {
        return this.nativeGraphHandle;
    }

    public synchronized void setStepMode(boolean stepMode) {
        this.stepMode = stepMode;
    }

    public synchronized boolean getStepMode() {
        return this.stepMode;
    }

    public synchronized void loadBinaryGraph(String path) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        nativeLoadBinaryGraph(this.nativeGraphHandle, path);
    }

    public synchronized void loadBinaryGraph(byte[] data) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        nativeLoadBinaryGraphBytes(this.nativeGraphHandle, data);
    }

    public synchronized void loadBinaryGraph(CalculatorProto.CalculatorGraphConfig config) {
        loadBinaryGraph(config.toByteArray());
    }

    public synchronized void loadBinaryGraphTemplate(GraphTemplateProto.CalculatorGraphTemplate template) {
        nativeLoadBinaryGraphTemplate(this.nativeGraphHandle, template.toByteArray());
    }

    public synchronized void setGraphType(String graphType) {
        nativeSetGraphType(this.nativeGraphHandle, graphType);
    }

    public synchronized void setGraphOptions(CalculatorProto.CalculatorGraphConfig.Node options) {
        nativeSetGraphOptions(this.nativeGraphHandle, options.toByteArray());
    }

    public synchronized CalculatorProto.CalculatorGraphConfig getCalculatorGraphConfig() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        byte[] data = nativeGetCalculatorGraphConfig(this.nativeGraphHandle);
        if (data != null) {
            try {
                return CalculatorProto.CalculatorGraphConfig.parseFrom(data);
            } catch (InvalidProtocolBufferException e2) {
                throw new RuntimeException(e2);
            }
        }
        return null;
    }

    public synchronized void addPacketCallback(String streamName, PacketCallback callback) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        Preconditions.checkNotNull(streamName);
        Preconditions.checkNotNull(callback);
        Preconditions.checkState((this.graphRunning || this.startRunningGraphCalled) ? false : true);
        this.packetCallbacks.add(callback);
        nativeAddPacketCallback(this.nativeGraphHandle, streamName, callback);
    }

    public synchronized SurfaceOutput addSurfaceOutput(String streamName) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        Preconditions.checkNotNull(streamName);
        Preconditions.checkState((this.graphRunning || this.startRunningGraphCalled) ? false : true);
        return new SurfaceOutput(this, Packet.create(nativeAddSurfaceOutput(this.nativeGraphHandle, streamName)));
    }

    public synchronized void setInputSidePackets(Map<String, Packet> sidePackets) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        Preconditions.checkState((this.graphRunning || this.startRunningGraphCalled) ? false : true);
        for (Map.Entry<String, Packet> entry : sidePackets.entrySet()) {
            this.sidePackets.put(entry.getKey(), entry.getValue().copy());
        }
    }

    public synchronized <T> void setServiceObject(GraphService<T> service, T object) {
        service.installServiceObject(this.nativeGraphHandle, object);
    }

    public synchronized void addStreamNameExpectingHeader(String streamName) {
        Preconditions.checkState((this.graphRunning || this.startRunningGraphCalled) ? false : true);
        this.streamHeaders.put(streamName, null);
    }

    public synchronized void setStreamHeader(String streamName, Packet streamHeader) {
        setStreamHeader(streamName, streamHeader, false);
    }

    public synchronized void setStreamHeader(String streamName, Packet streamHeader, boolean override) {
        Packet header = this.streamHeaders.get(streamName);
        if (header != null) {
            if (override) {
                if (this.graphRunning) {
                    throw new IllegalArgumentException("Can't override an existing stream header, after graph started running.");
                }
                header.release();
            } else {
                return;
            }
        }
        this.streamHeaders.put(streamName, streamHeader.copy());
        if (!this.graphRunning && this.startRunningGraphCalled && hasAllStreamHeaders()) {
            startRunningGraph();
        }
    }

    public synchronized void runGraphUntilClose() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        Preconditions.checkNotNull(this.sidePackets);
        String[] streamNames = new String[this.sidePackets.size()];
        long[] packets = new long[this.sidePackets.size()];
        splitStreamNamePacketMap(this.sidePackets, streamNames, packets);
        nativeRunGraphUntilClose(this.nativeGraphHandle, streamNames, packets);
    }

    public synchronized void startRunningGraph() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        this.startRunningGraphCalled = true;
        if (!hasAllStreamHeaders()) {
            logger.atInfo().log("MediaPipe graph won't start until all stream headers are available.");
            return;
        }
        String[] sidePacketNames = new String[this.sidePackets.size()];
        long[] sidePacketHandles = new long[this.sidePackets.size()];
        splitStreamNamePacketMap(this.sidePackets, sidePacketNames, sidePacketHandles);
        String[] streamNamesWithHeader = new String[this.streamHeaders.size()];
        long[] streamHeaderHandles = new long[this.streamHeaders.size()];
        splitStreamNamePacketMap(this.streamHeaders, streamNamesWithHeader, streamHeaderHandles);
        nativeStartRunningGraph(this.nativeGraphHandle, sidePacketNames, sidePacketHandles, streamNamesWithHeader, streamHeaderHandles);
        this.graphRunning = true;
        moveBufferedPacketsToInputStream();
    }

    public synchronized void setGraphInputStreamBlockingMode(boolean mode) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        Preconditions.checkState(!this.graphRunning);
        nativeSetGraphInputStreamBlockingMode(this.nativeGraphHandle, mode);
    }

    public synchronized void addPacketToInputStream(String streamName, Packet packet, long timestamp) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        if (!this.graphRunning) {
            addPacketToBuffer(streamName, packet.copy(), timestamp);
        } else {
            nativeAddPacketToInputStream(this.nativeGraphHandle, streamName, packet.getNativeHandle(), timestamp);
        }
    }

    public synchronized void addConsumablePacketToInputStream(String streamName, Packet packet, long timestamp) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        if (!this.graphRunning) {
            addPacketToBuffer(streamName, packet.copy(), timestamp);
            packet.release();
            return;
        }
        nativeMovePacketToInputStream(this.nativeGraphHandle, streamName, packet.getNativeHandle(), timestamp);
        packet.release();
    }

    public synchronized void closeInputStream(String streamName) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        nativeCloseInputStream(this.nativeGraphHandle, streamName);
    }

    public synchronized void closeAllInputStreams() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        nativeCloseAllInputStreams(this.nativeGraphHandle);
    }

    public synchronized void closeAllPacketSources() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        nativeCloseAllPacketSources(this.nativeGraphHandle);
    }

    public synchronized void waitUntilGraphDone() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        nativeWaitUntilGraphDone(this.nativeGraphHandle);
    }

    public synchronized void waitUntilGraphIdle() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
        nativeWaitUntilGraphIdle(this.nativeGraphHandle);
    }

    public synchronized void tearDown() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        for (Map.Entry<String, Packet> entry : this.sidePackets.entrySet()) {
            entry.getValue().release();
        }
        this.sidePackets.clear();
        for (Map.Entry<String, Packet> entry2 : this.streamHeaders.entrySet()) {
            if (entry2.getValue() != null) {
                entry2.getValue().release();
            }
        }
        this.streamHeaders.clear();
        for (Map.Entry<String, ArrayList<PacketBufferItem>> entry3 : this.packetBuffers.entrySet()) {
            Iterator<PacketBufferItem> it = entry3.getValue().iterator();
            while (it.hasNext()) {
                PacketBufferItem item = it.next();
                item.packet.release();
            }
        }
        this.packetBuffers.clear();
        synchronized (this.terminationLock) {
            if (this.nativeGraphHandle != 0) {
                nativeReleaseGraph(this.nativeGraphHandle);
                this.nativeGraphHandle = 0L;
            }
        }
        this.packetCallbacks.clear();
    }

    public synchronized void updatePacketReference(Packet referencePacket, Packet newPacket) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        nativeUpdatePacketReference(referencePacket.getNativeHandle(), newPacket.getNativeHandle());
    }

    @Deprecated
    public synchronized void createGlRunner(String name, long javaGlContext) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        Preconditions.checkArgument(name.equals("gpu_shared"));
        setParentGlContext(javaGlContext);
    }

    public synchronized void setParentGlContext(long javaGlContext) {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        Preconditions.checkState(!this.graphRunning);
        nativeSetParentGlContext(this.nativeGraphHandle, javaGlContext);
    }

    public synchronized void cancelGraph() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        nativeCancelGraph(this.nativeGraphHandle);
    }

    public GraphProfiler getProfiler() {
        Preconditions.checkState(this.nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
        return new GraphProfiler(nativeGetProfiler(this.nativeGraphHandle), this);
    }

    private boolean addPacketToBuffer(String streamName, Packet packet, long timestamp) {
        if (!this.packetBuffers.containsKey(streamName)) {
            this.packetBuffers.put(streamName, new ArrayList<>());
        }
        List<PacketBufferItem> buffer = this.packetBuffers.get(streamName);
        if (buffer.size() > 20) {
            for (Map.Entry<String, Packet> entry : this.streamHeaders.entrySet()) {
                if (entry.getValue() == null) {
                    logger.atSevere().log("Stream: %s might be missing.", entry.getKey());
                }
            }
            throw new RuntimeException("Graph is not started because of missing streams");
        }
        buffer.add(new PacketBufferItem(packet, Long.valueOf(timestamp)));
        return true;
    }

    private void moveBufferedPacketsToInputStream() {
        if (!this.packetBuffers.isEmpty()) {
            for (Map.Entry<String, ArrayList<PacketBufferItem>> entry : this.packetBuffers.entrySet()) {
                Iterator<PacketBufferItem> it = entry.getValue().iterator();
                while (it.hasNext()) {
                    PacketBufferItem item = it.next();
                    try {
                        nativeMovePacketToInputStream(this.nativeGraphHandle, entry.getKey(), item.packet.getNativeHandle(), item.timestamp.longValue());
                        item.packet.release();
                    } catch (MediaPipeException e2) {
                        logger.atSevere().log("AddPacket for stream: %s failed: %s.", entry.getKey(), e2.getMessage());
                        throw e2;
                    }
                }
            }
            this.packetBuffers.clear();
        }
    }

    private static void splitStreamNamePacketMap(Map<String, Packet> namePacketMap, String[] streamNames, long[] packets) {
        if (namePacketMap.size() != streamNames.length || namePacketMap.size() != packets.length) {
            throw new RuntimeException("Input array length doesn't match the map size!");
        }
        int i = 0;
        for (Map.Entry<String, Packet> entry : namePacketMap.entrySet()) {
            streamNames[i] = entry.getKey();
            packets[i] = entry.getValue().getNativeHandle();
            i++;
        }
    }

    private boolean hasAllStreamHeaders() {
        for (Map.Entry<String, Packet> entry : this.streamHeaders.entrySet()) {
            if (entry.getValue() == null) {
                return false;
            }
        }
        return true;
    }
}
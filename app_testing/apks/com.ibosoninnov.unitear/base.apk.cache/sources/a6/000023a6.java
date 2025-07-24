package com.google.common.hash;

import c.b.a.a.a;
import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.Immutable;
import java.io.Serializable;
import java.nio.ByteBuffer;

@Immutable
/* loaded from: classes.dex */
public final class SipHashFunction extends AbstractHashFunction implements Serializable {
    public static final HashFunction SIP_HASH_24 = new SipHashFunction(2, 4, 506097522914230528L, 1084818905618843912L);
    private static final long serialVersionUID = 0;

    /* renamed from: c  reason: collision with root package name */
    private final int f5635c;

    /* renamed from: d  reason: collision with root package name */
    private final int f5636d;
    private final long k0;
    private final long k1;

    /* loaded from: classes.dex */
    public static final class SipHasher extends AbstractStreamingHasher {
        private static final int CHUNK_SIZE = 8;

        /* renamed from: b  reason: collision with root package name */
        private long f5637b;

        /* renamed from: c  reason: collision with root package name */
        private final int f5638c;

        /* renamed from: d  reason: collision with root package name */
        private final int f5639d;
        private long finalM;
        private long v0;
        private long v1;
        private long v2;
        private long v3;

        public SipHasher(int i, int i2, long j, long j2) {
            super(8);
            this.v0 = 8317987319222330741L;
            this.v1 = 7237128888997146477L;
            this.v2 = 7816392313619706465L;
            this.v3 = 8387220255154660723L;
            this.f5637b = 0L;
            this.finalM = 0L;
            this.f5638c = i;
            this.f5639d = i2;
            this.v0 = 8317987319222330741L ^ j;
            this.v1 = 7237128888997146477L ^ j2;
            this.v2 = 7816392313619706465L ^ j;
            this.v3 = 8387220255154660723L ^ j2;
        }

        private void processM(long j) {
            this.v3 ^= j;
            sipRound(this.f5638c);
            this.v0 = j ^ this.v0;
        }

        private void sipRound(int i) {
            for (int i2 = 0; i2 < i; i2++) {
                long j = this.v0;
                long j2 = this.v1;
                this.v0 = j + j2;
                this.v2 += this.v3;
                this.v1 = Long.rotateLeft(j2, 13);
                long rotateLeft = Long.rotateLeft(this.v3, 16);
                this.v3 = rotateLeft;
                long j3 = this.v1;
                long j4 = this.v0;
                this.v1 = j3 ^ j4;
                this.v3 = rotateLeft ^ this.v2;
                long rotateLeft2 = Long.rotateLeft(j4, 32);
                this.v0 = rotateLeft2;
                long j5 = this.v2;
                long j6 = this.v1;
                this.v2 = j5 + j6;
                this.v0 = rotateLeft2 + this.v3;
                this.v1 = Long.rotateLeft(j6, 17);
                long rotateLeft3 = Long.rotateLeft(this.v3, 21);
                this.v3 = rotateLeft3;
                long j7 = this.v1;
                long j8 = this.v2;
                this.v1 = j7 ^ j8;
                this.v3 = rotateLeft3 ^ this.v0;
                this.v2 = Long.rotateLeft(j8, 32);
            }
        }

        @Override // com.google.common.hash.AbstractStreamingHasher
        public HashCode makeHash() {
            long j = this.finalM ^ (this.f5637b << 56);
            this.finalM = j;
            processM(j);
            this.v2 ^= 255;
            sipRound(this.f5639d);
            return HashCode.fromLong(((this.v0 ^ this.v1) ^ this.v2) ^ this.v3);
        }

        @Override // com.google.common.hash.AbstractStreamingHasher
        public void process(ByteBuffer byteBuffer) {
            this.f5637b += 8;
            processM(byteBuffer.getLong());
        }

        @Override // com.google.common.hash.AbstractStreamingHasher
        public void processRemaining(ByteBuffer byteBuffer) {
            this.f5637b += byteBuffer.remaining();
            int i = 0;
            while (byteBuffer.hasRemaining()) {
                this.finalM ^= (byteBuffer.get() & 255) << i;
                i += 8;
            }
        }
    }

    public SipHashFunction(int i, int i2, long j, long j2) {
        Preconditions.checkArgument(i > 0, "The number of SipRound iterations (c=%s) during Compression must be positive.", i);
        Preconditions.checkArgument(i2 > 0, "The number of SipRound iterations (d=%s) during Finalization must be positive.", i2);
        this.f5635c = i;
        this.f5636d = i2;
        this.k0 = j;
        this.k1 = j2;
    }

    @Override // com.google.common.hash.HashFunction
    public int bits() {
        return 64;
    }

    public boolean equals(Object obj) {
        if (obj instanceof SipHashFunction) {
            SipHashFunction sipHashFunction = (SipHashFunction) obj;
            return this.f5635c == sipHashFunction.f5635c && this.f5636d == sipHashFunction.f5636d && this.k0 == sipHashFunction.k0 && this.k1 == sipHashFunction.k1;
        }
        return false;
    }

    public int hashCode() {
        return (int) ((((SipHashFunction.class.hashCode() ^ this.f5635c) ^ this.f5636d) ^ this.k0) ^ this.k1);
    }

    @Override // com.google.common.hash.HashFunction
    public Hasher newHasher() {
        return new SipHasher(this.f5635c, this.f5636d, this.k0, this.k1);
    }

    public String toString() {
        StringBuilder x = a.x("Hashing.sipHash");
        x.append(this.f5635c);
        x.append("");
        x.append(this.f5636d);
        x.append("(");
        x.append(this.k0);
        x.append(", ");
        x.append(this.k1);
        x.append(")");
        return x.toString();
    }
}
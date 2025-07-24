package com.google.android.gms.internal.measurement;

import com.google.android.gms.internal.measurement.zzkb;
import com.google.android.gms.internal.measurement.zzkf;
import java.io.IOException;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public class zzkb<MessageType extends zzkf<MessageType, BuilderType>, BuilderType extends zzkb<MessageType, BuilderType>> extends zzin<MessageType, BuilderType> {
    public zzkf zza;
    private final zzkf zzb;

    public zzkb(MessageType messagetype) {
        this.zzb = messagetype;
        if (!messagetype.zzbO()) {
            this.zza = messagetype.zzbA();
            return;
        }
        throw new IllegalArgumentException("Default instance must be immutable.");
    }

    private static void zza(Object obj, Object obj2) {
        zzlu.zza().zzb(obj.getClass()).zzg(obj, obj2);
    }

    public final zzkb zzaA(zzkf zzkfVar) {
        if (!this.zzb.equals(zzkfVar)) {
            if (!this.zza.zzbO()) {
                zzaH();
            }
            zza(this.zza, zzkfVar);
        }
        return this;
    }

    public final zzkb zzaB(byte[] bArr, int i, int i2, zzjr zzjrVar) {
        if (!this.zza.zzbO()) {
            zzaH();
        }
        try {
            zzlu.zza().zzb(this.zza.getClass()).zzh(this.zza, bArr, 0, i2, new zzir(zzjrVar));
            return this;
        } catch (zzkp e2) {
            throw e2;
        } catch (IOException e3) {
            throw new RuntimeException("Reading from byte array should not throw IOException.", e3);
        } catch (IndexOutOfBoundsException unused) {
            throw zzkp.zzf();
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x002e, code lost:
        if (r3 != false) goto L13;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final MessageType zzaC() {
        MessageType zzaE = zzaE();
        byte byteValue = ((Byte) zzaE.zzl(1, null, null)).byteValue();
        if (byteValue != 1) {
            if (byteValue != 0) {
                boolean zzk = zzlu.zza().zzb(zzaE.getClass()).zzk(zzaE);
                zzaE.zzl(2, true != zzk ? null : zzaE, null);
            }
            throw new zzmn(zzaE);
        }
        return zzaE;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.android.gms.internal.measurement.zzll
    /* renamed from: zzaD */
    public MessageType zzaE() {
        if (this.zza.zzbO()) {
            this.zza.zzbJ();
            return (MessageType) this.zza;
        }
        return (MessageType) this.zza;
    }

    public final void zzaG() {
        if (this.zza.zzbO()) {
            return;
        }
        zzaH();
    }

    public void zzaH() {
        zzkf zzbA = this.zzb.zzbA();
        zza(zzbA, this.zza);
        this.zza = zzbA;
    }

    @Override // com.google.android.gms.internal.measurement.zzin
    public final /* bridge */ /* synthetic */ zzin zzav(byte[] bArr, int i, int i2) {
        zzaB(bArr, 0, i2, zzjr.zza);
        return this;
    }

    @Override // com.google.android.gms.internal.measurement.zzin
    public final /* bridge */ /* synthetic */ zzin zzaw(byte[] bArr, int i, int i2, zzjr zzjrVar) {
        zzaB(bArr, 0, i2, zzjrVar);
        return this;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // com.google.android.gms.internal.measurement.zzin
    /* renamed from: zzaz */
    public final zzkb zzau() {
        zzkb zzkbVar = (zzkb) this.zzb.zzl(5, null, null);
        zzkbVar.zza = zzaE();
        return zzkbVar;
    }

    @Override // com.google.android.gms.internal.measurement.zzln
    public final /* bridge */ /* synthetic */ zzlm zzbS() {
        throw null;
    }
}
package com.google.android.gms.internal.measurement;

import c.b.a.a.a;
import com.google.android.gms.internal.measurement.zzin;
import com.google.android.gms.internal.measurement.zzio;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement-base@@21.2.0 */
/* loaded from: classes.dex */
public abstract class zzio<MessageType extends zzio<MessageType, BuilderType>, BuilderType extends zzin<MessageType, BuilderType>> implements zzlm {
    public int zzb = 0;

    public static void zzbt(Iterable iterable, List list) {
        zzkn.zze(iterable);
        if (iterable instanceof zzku) {
            List zzh = ((zzku) iterable).zzh();
            zzku zzkuVar = (zzku) list;
            int size = list.size();
            for (Object obj : zzh) {
                if (obj == null) {
                    int size2 = zzkuVar.size();
                    StringBuilder x = a.x("Element at index ");
                    x.append(size2 - size);
                    x.append(" is null.");
                    String sb = x.toString();
                    int size3 = zzkuVar.size();
                    while (true) {
                        size3--;
                        if (size3 < size) {
                            break;
                        }
                        zzkuVar.remove(size3);
                    }
                    throw new NullPointerException(sb);
                } else if (obj instanceof zzje) {
                    zzkuVar.zzi((zzje) obj);
                } else {
                    zzkuVar.add((String) obj);
                }
            }
        } else if (!(iterable instanceof zzlt)) {
            if ((list instanceof ArrayList) && (iterable instanceof Collection)) {
                ((ArrayList) list).ensureCapacity(((Collection) iterable).size() + list.size());
            }
            int size4 = list.size();
            for (Object obj2 : iterable) {
                if (obj2 == null) {
                    int size5 = list.size();
                    StringBuilder x2 = a.x("Element at index ");
                    x2.append(size5 - size4);
                    x2.append(" is null.");
                    String sb2 = x2.toString();
                    int size6 = list.size();
                    while (true) {
                        size6--;
                        if (size6 < size4) {
                            break;
                        }
                        list.remove(size6);
                    }
                    throw new NullPointerException(sb2);
                }
                list.add(obj2);
            }
        } else {
            list.addAll((Collection) iterable);
        }
    }

    public int zzbr(zzlx zzlxVar) {
        throw null;
    }

    @Override // com.google.android.gms.internal.measurement.zzlm
    public final zzje zzbs() {
        try {
            int zzbw = zzbw();
            zzje zzjeVar = zzje.zzb;
            byte[] bArr = new byte[zzbw];
            zzjm zzC = zzjm.zzC(bArr);
            zzbN(zzC);
            zzC.zzD();
            return new zzjb(bArr);
        } catch (IOException e2) {
            throw new RuntimeException(a.r("Serializing ", getClass().getName(), " to a ByteString threw an IOException (should never happen)."), e2);
        }
    }

    public final byte[] zzbu() {
        try {
            byte[] bArr = new byte[zzbw()];
            zzjm zzC = zzjm.zzC(bArr);
            zzbN(zzC);
            zzC.zzD();
            return bArr;
        } catch (IOException e2) {
            throw new RuntimeException(a.r("Serializing ", getClass().getName(), " to a byte array threw an IOException (should never happen)."), e2);
        }
    }
}
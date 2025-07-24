package androidx.media;

import b.b0.a;
import java.util.Objects;

/* loaded from: classes.dex */
public final class AudioAttributesImplBaseParcelizer {
    public static AudioAttributesImplBase read(a aVar) {
        AudioAttributesImplBase audioAttributesImplBase = new AudioAttributesImplBase();
        audioAttributesImplBase.f338a = aVar.k(audioAttributesImplBase.f338a, 1);
        audioAttributesImplBase.f339b = aVar.k(audioAttributesImplBase.f339b, 2);
        audioAttributesImplBase.f340c = aVar.k(audioAttributesImplBase.f340c, 3);
        audioAttributesImplBase.f341d = aVar.k(audioAttributesImplBase.f341d, 4);
        return audioAttributesImplBase;
    }

    public static void write(AudioAttributesImplBase audioAttributesImplBase, a aVar) {
        Objects.requireNonNull(aVar);
        int i = audioAttributesImplBase.f338a;
        aVar.p(1);
        aVar.t(i);
        int i2 = audioAttributesImplBase.f339b;
        aVar.p(2);
        aVar.t(i2);
        int i3 = audioAttributesImplBase.f340c;
        aVar.p(3);
        aVar.t(i3);
        int i4 = audioAttributesImplBase.f341d;
        aVar.p(4);
        aVar.t(i4);
    }
}
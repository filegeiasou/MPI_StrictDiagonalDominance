# MPI_StrictDiagonalDominance

Ένας πίνακας Α(NxN) λέγεται αυστηρά διαγώνια δεσπόζων (strictly diagonally
dominant) εάν για κάθε γραμμή του πίνακα του Α ισχύει:
- 

Σας ζητείται να γράψετε και να τρέξετε ένα MPI πρόγραμμα σε C (θεωρώντας ένα
παράλληλο περιβάλλον p επεξεργαστών), το οποίο δοθέντος ενός δισδιάστατου
πίνακα Α(ΝxN):
- Α. Θα ελέγχει αρχικά με παράλληλο τρόπο αν ο πίνακας Α είναι αυστηρά διαγώνια
δεσπόζων (ο επεξεργαστής ‘0’ θα πρέπει να διαβάζει από την οθόνη τον πίνακα Α και
στο τέλος να τυπώνει το αποτέλεσμα – ‘yes’ ή ‘no’).
- Β. Στην περίπτωση που αυτό ισχύει (είναι δηλ. ο πίνακας Α αυστηρά διαγώνια
δεσπόζων) το πρόγραμμα θα πρέπει στη συνέχεια να υπολογίζει παράλληλα το
μέγιστο κατ’ απόλυτη τιμή στοιχείο της διαγωνίου του πίνακα Α (m=max(|Aii|).
- Γ. Kαι ακολούθως με βάση αυτό (m) να φτιάχνει παράλληλα ένα νέο πίνακα Β ΝxN
(τον οποίον θα τυπώνει επίσης ο '0' στο τέλος στην οθόνη) όπου:
   - Bij = m – |Aij| για i<>j και Bij = m για i=j
- Δ. Για τον παραπάνω πίνακα Β ζητείται επίσης να υπολογιστεί παράλληλα (και να
τυπώνεται στο τέλος από τον ‘0’ επίσης στην οθόνη) το ελάχιστο σε τιμή στοιχείο
του, καθώς και σε ποιά θέση (i,j) του πίνακα Β βρίσκεται.

Θεωρείστε ότι το ‘N’ είναι ακέραιο πολλαπλάσιο του ‘p’. Χρησιμοποιήστε στο
πρόγραμμά σας μόνο συναρτήσεις συλλογικής επικοινωνίας.
Το σύνολο του απαιτούμενου υπολογιστικού φόρτου θα πρέπει να ισοκατανεμηθεί
κατά το δυνατόν στους ‘p’ επεξεργαστές του παράλληλου περιβάλλοντός σας.
Επίσης, κάθε επεξεργαστής θα πρέπει να λαμβάνει (κατέχει) στην τοπική του μνήμη
μόνο τα δεδομένα εισόδου που χρησιμοποιεί για τοπικούς (δικούς του) υπολογισμούς.
Περιγράψτε επίσης (δεν ζητείται να το υλοποιήσετε παρά μόνο να το περιγράψετε),
πως θα επεκτείνατε το πρόγραμμά σας έτσι ώστε να συμπεριφέρεται σωστά για
οποιονδήποτε συνδυασμό τιμών ‘Ν’ και ‘p’
